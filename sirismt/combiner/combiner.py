import sys
sys.path.append('../..')
import math

import z3

from sirismt.combiner.solve_cache import *
from sirismt.tools import objects
from sirismt.tools.transformer import *
from sirismt.tools.types import *


def calc_best_probe(data, probe_dict, data_dict, predicts=None):
    if predicts is None: predicts = gen_predicts(probe_dict)
    count_dicts = ({}, {})
    best_pb = None
    max_ratio = -1
    for pb_i, pb in enumerate(predicts):
        data_true, data_false = split_data(data, pb, probe_dict)
        if data_true == [] or data_false == []:
            continue
        true_dict, undef_true = data_dict.get_tac_counts(data_true)
        false_dict, undef_false = data_dict.get_tac_counts(data_false)

        gain_ratio = calc_gain_ratio(true_dict, false_dict, pb, data_true, data_false, probe_dict)
        if gain_ratio > max_ratio:
            max_ratio = gain_ratio
            best_pb = pb
            count_dicts = true_dict, false_dict

    return best_pb, count_dicts


def _gen_increasing_arrays(num_items, target_sum, start=2):
    def _helper(target_, length_, start_):
        if target_ == 0 and length_ == 0:
            yield []
        elif target_ > 0 and length_ > 0:
            for i in range(start_, target_ + 1):
                for sub_array in _helper(target_ - i, length_ - 1, i):
                    yield [i] + sub_array

    for length in range(1, num_items+1):
        yield from _helper(target_sum, length, start)


def gen_predicts(probe_dict: dict[str, ProbeDict]) -> list[objects.ProbeCond]:
    """
    generate proper predicts with patterns (> probe value) for given data.
    """
    probe_set_dict = {p: set() for p in z3.probes()}

    for name, pd in probe_dict.items():
        for p, v in pd.items():
            probe_set_dict[p].add(v)

    predicts = []
    for probe, values in probe_set_dict.items():
        if probe not in ['num-consts', 'num-exprs', 'size']:
            continue
        if len(values) <= 1:
            continue
        p = objects.Probe(probe)
        values = list(set(values))
        values.sort()
        proper_values = []
        # for probe_i in range(len(values) - 1):
        #     value = (values[probe_i] + values[probe_i + 1]) / 2
        values.sort()
        n = 20
        candidate_n = [1/n * i for i in range(1, n)]
        for pos in candidate_n:
            proper_values.append(int(values[int(pos*len(values))]))
        proper_values = set(proper_values)
        predicts.extend([objects.ProbeCond(p, pv) for pv in proper_values])
    return predicts


def split_data(data: list[str], probe: objects.ProbeCond, probe_dict: dict[str, ProbeDict])\
        -> tuple[list[str], list[str]]:
    """
    split the given data using the given probe.
    :return: two list corresponded data which pass the probe or not.
    """
    data_true, data_false = [], []
    p = probe.s
    v = probe.cond
    for name in data:
        if probe_dict[name][p] > v:
            data_true.append(name)
        else:
            data_false.append(name)
    return data_true, data_false


def calc_iv(data, pb: objects.ProbeCond, probe_dict: dict):
    data_map = {}
    for data_name in data:
        if data_name in probe_dict and pb.s in probe_dict[data_name]:
            if probe_dict[data_name][pb.s] not in data_map:
                data_map[probe_dict[data_name][pb.s]] = 0
            data_map[probe_dict[data_name][pb.s]] += 1
    res_iv = -sum([cnt / len(data) * math.log2(cnt / len(data)) for _, cnt in data_map.items()])
    return res_iv


def calc_entropy(count_dict):
    data_len = sum(count_dict.values())
    entropy = 0
    for cnt in count_dict.values():
        ratio = cnt / data_len
        entropy -= ratio * math.log2(ratio)
    return entropy


def calc_gain_ratio(true_dict, false_dict, pb, data_true, data_false, probe_dict):
    count_dict = {key: value for key, value in true_dict.items()}
    for key, value in false_dict.items():
        if key not in count_dict:
            count_dict[key] = 0
        count_dict[key] += value
    tot_len = len(data_false) + len(data_true)
    self_entropy = calc_entropy(count_dict)
    true_entropy = calc_entropy(true_dict) * (len(data_true) / tot_len)
    false_entropy = calc_entropy(false_dict) * (len(data_false) / tot_len)
    gain = self_entropy - (true_entropy + false_entropy)
    iv = calc_iv(data_true + data_false, pb, probe_dict)
    return gain / iv


def _tac_seq2str(tac_seq: list) -> str:
    strategy = objects.AndThen(*tac_seq) if len(tac_seq) > 1 else tac_seq[0]
    if isinstance(strategy, str):
        strategy = objects.Tactic(strategy)
    return str(strategy)


class C45Combiner:
    def __init__(self, timeout):
        super().__init__()
        self.solve_cache = CompleteSolveCache(timeout)
        self.trace = []
        self.MIN_DATA_LENGTH = 1

    def gen_strategy(self, data: list[str], tac_seqs: list[list[objects.Tactic]], cache_path: str = None,
                     batch_size: int = 8) -> objects.Tactic:
        tac_seqs = [(_tac_seq2str(tac_seq), tac_seq) for tac_seq in tac_seqs]
        self.solve_cache.init_batch(data, tac_seqs, cache_path, batch_size, 5)
        if data is None:
            data = list(self.solve_cache.cache.keys())
        return self.gen_strategy_dfs(data, [])

    def find_minimum_strategy(self, data, prefix, num_select=5, timeout=None) -> objects.Tactic:
        data_dict = self.solve_cache.create_data_dict(data, prefix)
        if timeout is None: timeout = self.solve_cache.timeout
        res, time_limit = [], []
        min_undef = len(data)
        for time_limit_ in _gen_increasing_arrays(num_select, timeout, math.ceil(timeout/10)):
            res_, undef = data_dict.get_par_tacs(data, time_limit_)
            if len(res_) != len(time_limit_): continue
            if undef >= min_undef:
                continue
            min_undef = undef
            res = res_
            time_limit = time_limit_

        if len(res) == 0: return objects.Tactic('skip')
        final_strategy = None
        for strategy, limit, in zip(res, time_limit):
            if final_strategy is None:
                final_strategy = strategy
                continue
            try_strategy = objects.TryFor(limit*1000, strategy)
            final_strategy = objects.OrElse(try_strategy, final_strategy)
        return final_strategy

    def _filter_data(self, data, prefix):
        return [instance for instance in data if len(self.solve_cache.get_tac_seq(instance, prefix)) > 0]

    def gen_strategy_forward(self, data, prefix, *args) -> tuple[list[str], list[objects.Tactic]]:
        data_dict = self.solve_cache.create_data_dict(data, prefix).simplify()
        new_tacs = []
        while (common_tac := data_dict.get_common_tac(data)) is not None:
            prefix.append(str(common_tac))
            new_tacs.append(common_tac)
            data_dict = self.solve_cache.create_data_dict(data, prefix).simplify()
        return self._filter_data(data, prefix), new_tacs

    def gen_strategy_if(self, data: list[str], prefix: list, predicts=None, par_time=None, *args):
        probe_dicts = {data_name: probe_dict for data_name in data
                       if (probe_dict := self.solve_cache.get_probes(data_name, prefix)) is not None}
        if predicts is None:
            predicts = gen_predicts(probe_dicts)
        n_data = list(probe_dicts.keys())
        best_pb, count_dicts = calc_best_probe(n_data, probe_dicts,
                                               self.solve_cache.create_data_dict(n_data, prefix), predicts)
        if best_pb is None:
            return self.find_minimum_strategy(data, prefix, timeout=par_time)
        data = [dt for dt in data if dt not in n_data]

        data_left, data_right = split_data(n_data, best_pb, probe_dicts)
        data_left.extend(data)
        data_right.extend(data)

        tac_left = self.gen_strategy_dfs(data_left, prefix.copy(), predicts, par_time)
        tac_right = self.gen_strategy_dfs(data_right, prefix.copy(), predicts, par_time)
        return objects.Cond(best_pb, tac_left, tac_right)

    def gen_strategy_or(self, data, prefix, timeout):
        tac, solve_num, timeout = self.solve_cache.create_data_dict(data, prefix).simplify().get_tac_dicts(data,
                                                                                                           timeout)
        if tac is None or solve_num < len(data) // 3:
            return None

        timeout = math.ceil(timeout)
        ano_prefix = prefix.copy()
        ano_prefix.append(str(tac))
        ano_data = self._filter_data(data, ano_prefix)
        tac_try = objects.TryFor(1000 * timeout,
                                 combine_strategy([tac],
                                                  self.gen_strategy_dfs(ano_data, ano_prefix, par_time=timeout)))
        ano_data = [dt for dt in data if dt not in ano_data]
        tac_ano = self.gen_strategy_dfs(ano_data, prefix.copy(), par_time=self.solve_cache.timeout - timeout)
        tac_or = objects.OrElse(tac_try, tac_ano)
        return tac_or

    def gen_strategy_dfs(self, data: list[str], prefix: list, predicts=None, par_time=1, *args) -> objects.Tactic:
        if len(data) == 0:
            return objects.Tactic('skip')
        data, now_tacs = self.gen_strategy_forward(data, prefix, *args)
        if len(now_tacs) > 0: predicts = None
        if len(data) < self.MIN_DATA_LENGTH:
            return combine_strategy(now_tacs, self.find_minimum_strategy(data, prefix, timeout=par_time))

        if par_time is not None:
            return combine_strategy(now_tacs, self.gen_strategy_if(data, prefix, predicts, par_time, *args))

        tac_or = self.gen_strategy_or(data, prefix, 1)
        if tac_or is not None:
            return combine_strategy(now_tacs, tac_or)

        return combine_strategy(now_tacs, self.gen_strategy_if(data, prefix, predicts, par_time, *args))


class C45CombinerX(C45Combiner):
    def __init__(self, timeout):
        super().__init__(timeout)

    def gen_strategy_forward(self, data, prefix, par_time=None,
                             max_depth=3, *args) -> tuple[list[str], list[objects.Tactic]]:
        if max_depth > 0:
            return super().gen_strategy_forward(data, prefix)
        now_tacs = []
        tac, solve_num, timeout = self.solve_cache.create_data_dict(data, prefix).simplify().get_tac_dicts(data, par_time)
        now_tacs.append(tac)
        prefix.append(str(tac))
        data = self._filter_data(data, prefix)
        data, ano_tacs = super().gen_strategy_forward(data, prefix)
        now_tacs.extend(ano_tacs)
        return data, now_tacs

    def gen_strategy_dfs(self, data: list[str], prefix: list, predicts=None, par_time=None,
                         max_depth=2, *args) -> objects.Tactic:
        if max_depth > 0 or len(data) < self.MIN_DATA_LENGTH:
            return super().gen_strategy_dfs(data, prefix, predicts, par_time, max_depth)

        data, now_tacs = self.gen_strategy_forward(data, prefix,
                                                   par_time if par_time is not None else self.solve_cache.timeout,
                                                   max_depth)
        return combine_strategy(now_tacs, self.gen_strategy_dfs(data, prefix, None, par_time, 10))

    def gen_strategy_if(self, data: list[str], prefix: list, predicts=None, par_time=None, max_depth=10, *args):
        probe_dicts = {data_name: probe_dict for data_name in data
                       if (probe_dict := self.solve_cache.get_probes(data_name, prefix)) is not None}
        if predicts is None: predicts = gen_predicts(probe_dicts)
        n_data = list(probe_dicts.keys())
        best_pb, count_dicts = calc_best_probe(n_data, probe_dicts,
                                               self.solve_cache.create_data_dict(n_data, prefix), predicts)
        if best_pb is None:
            return self.find_minimum_strategy(data, prefix, timeout=par_time)
        data = [dt for dt in data if dt not in n_data]

        data_left, data_right = split_data(n_data, best_pb, probe_dicts)
        data_left.extend(data)
        data_right.extend(data)

        tac_left = self.gen_strategy_dfs(data_left, prefix.copy(), predicts, par_time, max_depth - 1)
        tac_right = self.gen_strategy_dfs(data_right, prefix.copy(), predicts, par_time, max_depth - 1)
        return objects.Cond(best_pb, tac_left, tac_right)
