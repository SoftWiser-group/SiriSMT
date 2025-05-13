import sys
sys.path.append('../..')

import copy
import multiprocessing
import random

from sirismt.tools import objects
from sirismt.tools.solver import solve_batch_cross
from sirismt.combiner.prob_dd import iterative_prob_dd
from sirismt.combiner.statistic import collect_subseqs

from sirismt.tools.transformer import make_strategy
from sirismt.tools.types import *

__all__ = [
    'RandomTuner',
    'EmptyTuner',
    'refine_tac_group',
    'prob_dd_regroup'
]


def uniq_list_with_string(tac_seqs: list[TacticSeq]) -> list[TacticSeq]:
    """
    Unifies the list by determining whether the strings between neighbouring items are the same.

    Args:
        tac_seqs: Sorted list of lists.

    Returns:
        List[List[str]]: Unified list.
    """
    if len(tac_seqs) == 0:
        return tac_seqs.copy()
    res_seqs = [tac_seqs[0]]
    judge_string = str([str(tac) for tac in res_seqs[0]])
    for tac_seq in tac_seqs:
        feature_string = str([str(tac) for tac in tac_seq])
        if feature_string == judge_string:
            continue
        judge_string = feature_string
        res_seqs.append(tac_seq)
    return res_seqs


def _prob_dd_one_sequence(params):
    tac_seq, instances = params
    list_tac_seq = eval(tac_seq)
    list_tac_seq = iterative_prob_dd(list_tac_seq, instances)
    tac_seq = str(list_tac_seq)
    return tac_seq, instances


def prob_dd_regroup(tac_group: TacticSeqGroup) -> TacticSeqGroup:
    """
    Reduce the tactic sequence by applying probability delta debugging algorithm.

    This method may lead to different results cause by probability algorithm, but ensures the result will perform
    better in instance group.

    Args:
        tac_group: Tactic group for reducing.

    Returns:
        TacticSeqGroup: The TacticSeqGroup which has been reduced.

    """
    res_group = {}
    for tac_seq, instances in tac_group.items():
        list_tac_seq = eval(tac_seq)
        list_tac_seq = iterative_prob_dd(list_tac_seq, instances)
        tac_seq = str(list_tac_seq)
        if tac_seq not in res_group:
            res_group[tac_seq] = []
        res_group[tac_seq].extend(instances)
    return res_group


def group_list_by_instance(solved_tuples: list[CandidateSeq]) -> TacticSeqGroup:
    """
    Groups instance-tactic sequence tuples by tactic sequences.

    This method organizes the tuples by identifying the same tactic sequences and collects the corresponding
    instances into a single list. Tactic sequences that are empty, or start with 'None' are excluded from the grouping.

    Args:
        solved_tuples: A list of tuples, where each tuple contains an instance and a tactic sequence can solve it.

    Returns: Dict[UnchangedList,List[str]]: A dictionary with keys as tactic sequences and values as lists of
    instance that correspond to those tactic sequences.

    """
    res_dict = {}
    for instance, tac_seq in solved_tuples:
        if str(tac_seq) == 'None' or len(tac_seq) == 0 or str(tac_seq[0]) == 'None':
            continue
        tac_seq = str(tac_seq)
        if tac_seq not in res_dict:
            res_dict[tac_seq] = []
        res_dict[tac_seq].append(instance)
    return res_dict


def uniq_tac_seqs(tac_seqs: list[TacticSeq]) -> list[TacticSeq]:
    """
    Sort and unify the tactic sequences by judge the string of its contents.

    Args:
        tac_seqs: Sorted list of lists.

    Returns:
        List[List[str]]: Unified list.

    """
    tac_seqs.sort(key=lambda x: str([str(t) for t in x]))
    tac_seqs = uniq_list_with_string(tac_seqs)
    return tac_seqs


def _mutate_one_sequence(params):
    fis_tree, tac_seq, instances = params
    tac_seq = eval(tac_seq)
    mutate_seq = fis_tree.mutate(tac_seq, instances)
    mutate_seq = [str(n_seq) for n_seq in mutate_seq]
    return mutate_seq, instances


def mutate_group(tac_group: TacticSeqGroup, batch_size=4) -> TacticSeqGroup:
    """
    Mutate the tactic group by finding frequent item-sets and association rules of it.

    This method may lead to different results cause by real-time performance, but ensures the result will perform
    better in instance group.

    Args:
        batch_size:
        tac_group: Tactic group for mutating.

    Returns:
        TacticSeqGroup: The TacticSeqGroup include original items and mutated one.

    """
    tac_seqs = [tac for tac, _ in tac_group.items()]
    fis_tree = collect_subseqs(tac_seqs)
    res_group = {}
    with multiprocessing.Pool(processes=batch_size) as pool:
        results = pool.map(_mutate_one_sequence,
                           [(fis_tree, tac_seq, instances) for tac_seq, instances in tac_group.items()])
    for mutate_seq, instances in results:
        for n_seq in mutate_seq:
            if n_seq not in res_group:
                res_group[n_seq] = set()
            for instance in instances:
                res_group[n_seq].add(instance)
    mutated_group = {tac: list(instances) for tac, instances in res_group.items()}
    res_group = {}
    with multiprocessing.Pool(processes=batch_size) as pool:
        results = pool.map(_prob_dd_one_sequence,
                           [(tac_seq, instances) for tac_seq, instances in mutated_group.items()])
    for refined_seq, instances in results:
        if refined_seq not in res_group:
            res_group[refined_seq] = []
        res_group[refined_seq].extend(instances)

    return res_group


def merge_group(*groups: TacticSeqGroup) -> TacticSeqGroup:
    """
    Merge and uniq the different tactic groups into ones.

    Args:
        *groups: Tactic group for merging.

    Returns:
        TacticSeqGroup: The TacticSeqGroup which has been reduced.

    """
    res_group = {}
    for group in groups:
        for seq, instances in group.items():
            if seq not in res_group:
                res_group[seq] = set()
            res_group[seq].union(instances)
    return {seq: list[instance_set] for seq, instance_set in res_group.items()}


def _choose_best_tac_seqs(formula_costs, tac_seqs, shrink_size):
    formula_costs = copy.deepcopy(formula_costs)
    tac_cost = [0 for _ in range(len(tac_seqs))]
    for cost_list in formula_costs:
        tot_cost = sum(cost_list)
        for tac_i, cost in enumerate(cost_list):
            cost_list[tac_i] = cost / tot_cost

    chosen_tacs = []
    chosen_dict = {i: 1 for i in range(len(formula_costs))}
    while len(chosen_tacs) < shrink_size:
        for formula_i, cost_list in enumerate(formula_costs):
            for tac_i, cost in cost_list:
                tac_cost[tac_i] += cost / chosen_dict[formula_i]

        tac_cost.sort()
        chosen_tacs.append(tac_cost[0])
        for formula_i, cost_list in enumerate(formula_costs):
            if cost_list[tac_cost[0]] > 0:
                chosen_dict[formula_i] += 1

    return [tac_seqs[tac_i] for tac_i in chosen_tacs]


def shrink_tac_seqs(smt_instances: list[str], tac_seqs: list[TacticSeq],
                    shrink_size: int, batch_size: int = 8, timeout=10) -> list[TacticSeq]:
    """
    Shrink tactic sequences using small set of smt_instances.

    This method samples a small set from smt_instances, and evaluates the total cost of each tactic sequence at
    solving progress, keeps the smaller one.

    Args:
        batch_size: The batch for evaluating tactic sequence.
        smt_instances: A list of path to SMT file, which is for construct small valid set.
        tac_seqs: A list of tactic sequences.
        shrink_size: The optimal size of shrinking result.
        timeout: timeout.

    Returns:
        list[list[str]]: The shrunk list.

    """
    if len(tac_seqs) <= shrink_size:
        return tac_seqs
    small_set = random.sample(smt_instances, min(24, len(smt_instances)))

    tactics = [make_strategy(*tac_seq) for tac_seq in tac_seqs]
    res_list = solve_batch_cross(small_set, tactics, timeout, batch_size)
    formula_cost = [[max(timeout-cost, 0) for cost in time_cost] for time_cost in res_list]

    # for seq_i, tac_seq in enumerate(tqdm.tqdm(tac_seqs, desc="shrink")):
    #     tactic = make_strategy(*tac_seq)
    #     time_cost, _ = solver.solve_batch_same_tactic(small_set, tactic, batch_size=batch_size)
    #     time_cost = [max(time_cost) - rtime for rtime in time_cost]
    #     formula_cost.append(time_cost)

    formula_cost = [[formula_cost[i][k] for i in range(len(tac_seqs))] for k in range(len(small_set))]
    tac_cost = [0 for _ in range(len(tac_seqs))]
    for cost_list in formula_cost:
        tot_cost = sum(cost_list) + 1e-10
        for tac_i, cost in enumerate(cost_list):
            tac_cost[tac_i] += cost / tot_cost
    tac_cost = [(cost, tac_i) for tac_i, cost in enumerate(tac_cost)]
    tac_cost.sort()
    return [tac_seqs[tac_i] for _, tac_i in tac_cost[len(tac_seqs) - shrink_size:]]


def refine_tac_group(smt_instances: list[str], tac_seqs, fis_tac_seqs: list = None) -> list[list[str]]:
    tac_group = group_list_by_instance(list(zip(smt_instances, tac_seqs)))
    tac_group = prob_dd_regroup(tac_group)

    tac_seqs = [eval(tac) for tac, _ in tac_group.items()]
    tac_seqs = uniq_tac_seqs(tac_seqs)
    if fis_tac_seqs is None:
        fis_tac_seqs = tac_seqs
    fis_tac_seqs = [str(f_seq) for f_seq in fis_tac_seqs]
    fis_tree = collect_subseqs(fis_tac_seqs)
    if fis_tree is None:
        return tac_seqs
    n_tac_group = {}
    for tac_seq, instances in tac_group.items():
        n_tac_seqs = fis_tree.mutate(eval(tac_seq), instances)
        n_tac_seqs = [str(n_seq) for n_seq in n_tac_seqs]
        for n_seq in n_tac_seqs:
            if n_seq not in n_tac_group:
                n_tac_group[n_seq] = set()
            for instance in instances:
                n_tac_group[n_seq].add(instance)
    n_tac_group = {tac: list(insts) for tac, insts in n_tac_group.items()}
    n_tac_group = prob_dd_regroup(n_tac_group)
    tac_seqs.extend([eval(tac) for tac, _ in n_tac_group.items()])
    tac_seqs = uniq_tac_seqs(tac_seqs)
    return tac_seqs


class EmptyTuner:
    """This Class implements refinement stage, but do nothing at tuning."""

    def __init__(self, config, enumerator):
        self.config = config
        self.enumerator = enumerator

    def tune_groups(self, tac_group: TacticSeqGroup) -> list[list[str | objects.Tactic]]:
        """
        Tuning the tactic sequence in tactic groups.

        This method do nothing to original tactic sequences.

        Args:
            tac_group: Tactic group for tuning.

        Returns:
            list[list[str|Tactic]]: The tuned tactic sequences.

        """
        return [eval(tac_seq) for tac_seq in list(tac_group.keys())]

    def tuning(self, candidate_seqs: list[tuple[str, list]], batch_size=8) -> list[Strategy]:
        """tuning tactic sequences and return the best of them"""
        if not candidate_seqs:
            return []
        tac_group = group_list_by_instance(candidate_seqs)
        n_group = merge_group(mutate_group(tac_group, batch_size), tac_group)
        tac_seqs = [eval(seq) for seq, _ in tac_group.items()]

        smt_instances = [name for name, _ in candidate_seqs]
        tsp = self.tune_groups(n_group)
        tsp = uniq_tac_seqs(tsp)
        tsp = shrink_tac_seqs(smt_instances, tsp, self.config['shrink_size'], batch_size=batch_size)

        tsp.extend(tac_seqs)
        tsp = [make_strategy(*tac_seq) for tac_seq in tsp]
        with open(self.config['out_file'], 'w') as f:
            for st in tsp:
                f.write(f"{str(st)}\n")
        return tsp


class RandomTuner(EmptyTuner):
    """ This Class implements refinement stage, random parameters at tuning. """

    def __init__(self, config, enumerator):
        super().__init__(config, enumerator)

    def tune_groups(self, tac_group: dict) -> list:
        """
        Tuning the tactic sequence in tactic groups.

        This method sample random values for each params which is allowed to tune.

        Args:
            tac_group: Tactic group for tuning.

        Returns:
            list[list[str|Tactic]]: The tuned tactic sequences.

        """
        tsp = []
        tac_seqs = [eval(tac_seq) for tac_seq in list(tac_group.keys())]

        def tune_tactic(tactic: str | objects.Tactic) -> objects.Tactic:
            if isinstance(tactic, objects.Tactic):
                tactic = tactic.s
            args = {tac: random.random() for tac in self.enumerator.param_max[tactic]}
            final_tactic = self.enumerator.get_tactic_with_args(tactic, args)
            return final_tactic

        for seq in tac_seqs:
            new_seq = [tune_tactic(tac) for tac in seq]
            different_cnt = 0
            for x, nx in zip(seq, new_seq):
                different_cnt += str(x) != str(nx)
            if different_cnt == 0:
                continue
            tsp.extend([[tune_tactic(tac) for tac in seq] for _ in range(self.config['boost_num'])])

        return tsp
