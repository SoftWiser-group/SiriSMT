import sys
sys.path.append('../..')

from sirismt.tools.objects import *


class StrategyEnumerator:
    """ Class which is wrapper over all possible strategies. """

    def __init__(self, all_tactics: list, allowed_params: dict) -> None:
        self.all_tactics = all_tactics
        self.allowed_params = allowed_params
        self.base_tactics = [Tactic(tactic) for tactic in self.all_tactics]

        self.param_min = {}
        self.param_max = {}
        self.param_default = {}

        for tactic in self.all_tactics:
            self.param_min[tactic] = {}
            self.param_max[tactic] = {}
            self.param_default[tactic] = {}
            if tactic in self.allowed_params:
                if 'boolean' in self.allowed_params[tactic]:
                    for bool_param, default_value in self.allowed_params[tactic]['boolean'].items():
                        self.param_min[tactic][bool_param] = 0
                        self.param_max[tactic][bool_param] = 1
                        self.param_default[tactic][bool_param] = default_value
                if 'integer' in self.allowed_params[tactic]:
                    for int_param, value_range in self.allowed_params[tactic]['integer'].items():
                        min_value = value_range[0]
                        max_value = value_range[1]
                        self.param_min[tactic][int_param] = min_value
                        self.param_max[tactic][int_param] = max_value

    def get_tactic_with_args(self, tactic: str, args: dict[str, float]):
        if tactic not in self.allowed_params:
            return Tactic(tactic)

        params = {}

        for arg, value in args.items():
            true_value = self.param_min[tactic][arg] + value * (
                    self.param_max[tactic][arg] - self.param_min[tactic][arg])
            if (self.allowed_params[tactic].get('boolean') is not None) and (
                    arg in self.allowed_params[tactic]['boolean']):
                params[arg] = False if true_value < 0.5 else True
            else:
                params[arg] = int(true_value)

        with_tactic = With(tactic, params)
        return with_tactic
