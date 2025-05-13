import sys
sys.path.append('../..')

import z3
import math

import torch
from torch import nn

from sirismt.tools.solver import solve_with_timeout, get_probes
from sirismt.tools import objects
from sirismt.tools.transformer import make_strategy


__all__ = [
    'SMTEnv'
]

class PositionalEncoding(nn.Module):
    """This module implements the positional encoding."""
    def __init__(self, observation_state: int, dropout: float, max_len: int =500) -> None:
        """
        Args:
            observation_state: The number of features in the observation state.
            dropout: The dropout rate to be applied after adding the positional encoding.
            max_len: The maximum length of the sequence to be encoded.

        Returns:
            None
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, observation_state)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, observation_state, 2) * (math.log(10000.0) / observation_state))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)


class SMTEnv:
    """This class implement an environment interacting with z3-solver to solve SMT instance."""
    def __init__(self, all_tactics: list[str], with_position: bool=True) -> None:
        """
        Args:
            all_tactics: Allowed tactics to apply.
            with_position: Decide whether using position encoding.

        Returns:
            None
        """
        self.all_tactics = all_tactics
        self.state = None
        self.instance = None
        self.passed_action = []
        self.tried_action = set()

        self.tactic_layer = nn.Embedding(len(self.all_tactics) + 1, 10)
        self.pe_layer = PositionalEncoding(10, 0.2)
        self.with_position=with_position
        self.last_rtime = 0
        pass

    def _extract_actions(self) -> list[float]:
        """
        Encoding the tactic sequence have passed.

        This method calculate sequence's embedding by calculate the embedding of each tactic within sequence and add with position encoding of them.

        Returns:
            list: A list of floats representing the vector of the tactic sequence embedding.
        """
        tac_seq = [len(self.all_tactics)]
        tac_seq.extend(self.passed_action)
        embedded_seq = torch.as_tensor(tac_seq)
        embedded_vector = self.tactic_layer(embedded_seq)
        embedded_vector = self.pe_layer(embedded_vector)
        pooled_vector = torch.mean(embedded_vector, dim=0)[0]
        return pooled_vector.tolist()

    def reset(self, instance: str) -> None:
        """
        Reset the environment.

        This method reset the formula to initial state and empty the action message stored by env.

        Args:
            instance: The path to the SMT file which will be interacted

        Returns:
            None
        """
        self.instance = instance
        formula = z3.parse_smt2_file(instance)
        self.state = get_probes(formula)
        self.passed_action = []
        self.tried_action = set()
        self.last_rtime = 0

    def observe_state(self) -> list[float]:
        return self.state

    def is_tried(self, action: int) -> bool:
        return action in self.tried_action

    def is_full_try(self) -> bool:
        return len(self.tried_action) == len(self.all_tactics)

    def get_tac_seq(self) -> list[str]:
        return [self.all_tactics[i] for i in self.passed_action]

    def step(self, action: int) -> tuple[list[float], list[float], bool, float]:
        """
        Take an action to the current state, acquire the information of the process.

        Args:
            action: The action will be applied to current state.

        Returns:
            tuple: A tuple contains:
                - s (list[float]): The state before apply tactic.
                - s_ (list[float]): The state after apply tactic.
                - res (str): True if the formula has been solved by given tactic (False if any exception occurs).
                - cost (float): Par-2 cost of process, or -1 if failed.

        """
        tactic = self.all_tactics[action]

        s = self.state.copy()
        s_ = s.copy()
        if action in self.tried_action:
            return s, s.copy(), False, -1

        new_strategy = make_strategy(*self.get_tac_seq(), tactic)

        try:
            res, rtime, n_formula = solve_with_timeout(self.instance, new_strategy)
        except z3.z3types.Z3Exception:
            res, rtime, n_formula = 'unknown', 20, None

        if n_formula is not None:
            s_ = get_probes(n_formula)

        if s[:len(s_)] == s_ or n_formula is None:
            self.tried_action.add(action)
            return s, s.copy(), False, -1

        self.tried_action.clear()
        self.passed_action.append(action)
        rtime = rtime - self.last_rtime
        self.last_rtime += rtime
        if self.with_position:
            s_.extend(self._extract_actions())

        self.state = s_.copy()
        return s, s_, res in {'sat', 'unsat'}, max(rtime, 1e-5)
