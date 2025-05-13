import sys
import warnings

sys.path.append('../..')

import os.path
import time
from bisect import insort

from sirismt.tools.transformer import make_strategy, parse_strategy, strategy2list
from sirismt.tools.solver import solve_with_timeout, solve_batch_cross, solve_batch_use_runner
from sirismt.tools.types import *
from sirismt.tools import objects

from torch.multiprocessing import Pool
from multiprocessing import context

from tqdm import tqdm
import z3

__all__ = ['CompleteSolveCache']


def _fill_cache_path(cache_path: Optional[str], suffix: str) -> str:
    if cache_path is not None and cache_path != '':
        return cache_path
    time_str = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    return f'cache/{time_str}.{suffix}'


def _collect_probe(param_tuple: tuple[str, str, float]) -> Optional[dict[str, float]]:
    strategy, instance, timeout = param_tuple
    strategy = parse_strategy(strategy)
    _, _, formula = solve_with_timeout(instance, strategy, timeout)
    if formula is None:
        return None
    goal = z3.Goal()
    goal.add(formula)
    probes = [(p, z3.Probe(p)) for p in z3.probes()]

    probe_dict = {}
    for name, p in probes:
        probe_dict[name] = p(goal)
    return probe_dict


def _create_probe_dict(instance, tac_seq, timeout) -> dict[int, ProbeDict]:
    probe_dict = {}
    tac_seq = [objects.Tactic('skip')] + tac_seq
    pass_tactic = []

    res_list = []
    with Pool(10) as pool:
        for tac_i, tac in enumerate(tac_seq):
            pass_tactic.append(tac)
            res_list.append(pool.apply_async(_collect_probe, ((str(make_strategy(*pass_tactic)), instance, timeout),)))
        for res_i, res in enumerate(res_list):
            try:
                depth_dict = res.get(timeout=15)
            except context.TimeoutError:
                depth_dict = None
            if depth_dict is None:
                break
            probe_dict[res_i] = depth_dict
    return probe_dict



class CompleteDataDict:
    def __init__(self, data_dict, prefix, simplified=False):
        self.depth = len(prefix)
        self.item = data_dict
        self.simple = simplified
        self.prefix = prefix

    def simplify(self):
        new_res = {}
        for data_name, strategy_tuples in self.item.items():
            if len(strategy_tuples) == 0:
                continue
            new_res[data_name] = [(str(strategy2list(parse_strategy(strategy))[0]), weight)
                                  for strategy, weight in strategy_tuples]
        return CompleteDataDict(new_res, self.prefix, True)

    def _calc_tac_nums(self, tac_dicts: dict):
        res = {}
        for values in tac_dicts.values():
            for tac, _ in values:
                res[tac] = res.get(tac, 0) + 1
        res = list(res.items())
        res.sort(key=lambda x: x[1])
        return res[-1]

    def _calc_tac_times(self, tac_dicts: dict, time_limit):
        res = {}
        for values in tac_dicts.values():
            for tac, weight in values:
                if tac not in res: res[tac] = (0, 0)
                times_, rtimes_ = res[tac]
                res[tac] = (times_ + 1, max(rtimes_, weight))
        timeout_tac = [tac for tac, time_tuple in res.items() if time_tuple[1] > time_limit]
        for tac in timeout_tac:
            del res[tac]
        res = list(res.items())
        res.sort(key=lambda x: 100 * x[1][0] - x[1][1])
        return res[-1] if len(res) > 0 else (None, None)

    def get_tac_counts(self, data: list, force_best=False) -> tuple[dict, int]:
        if not self.simple: return self.simplify().get_tac_counts(data)

        tac_dicts = {data_name: tac_dict for data_name in data if len(tac_dict := self.item.get(data_name, [])) > 0}
        undef = len(data) - len(tac_dicts)
        res = {}

        if force_best:
            for _, values in tac_dicts.items():
                values = list(values)
                values.sort(key=lambda x: x[1])
                tac = values[0][0]
                res[tac] = res.get(tac, 0) + 1
            return res, undef

        while len(tac_dicts) > 0:
            nxt, num = self._calc_tac_nums(tac_dicts)
            res[nxt] = num
            removed_data = [data_name for data_name in tac_dicts.keys() if nxt not in tac_dicts[data_name]]
            for rm_data in removed_data:
                del tac_dicts[rm_data]

        return res, undef

    def get_tac_dicts(self, data: list, time_limit: int):
        if not self.simple: return self.simplify().get_tac_dicts(data, time_limit)
        tac_dicts = {data_name: tac_dict for data_name in data if len(tac_dict := self.item.get(data_name, [])) > 0}

        nxt, time_tuple = self._calc_tac_times(tac_dicts, time_limit)
        if nxt is not None:
            return parse_strategy(nxt), time_tuple[0], time_tuple[1]
        return None, None, None

    def get_common_tac(self, data: list) -> Optional[objects.Tactic]:
        if not self.simple: return self.simplify().get_common_tac(data)
        res_dict = None
        for data_name in data:
            tac_list = self.item.get(data_name, [])
            if len(tac_list) == 0:
                continue
            tac_dict = {tac: weight for tac, weight in tac_list}
            if res_dict is None:
                res_dict = tac_dict
                continue
            common_tac = set(tac_dict.keys()) & set(res_dict.keys())
            if len(common_tac) == 0:
                return None
            res_dict = {tac: res_dict[tac] + tac_dict[tac] for tac in common_tac}
        if res_dict is None:
            return None
        common_tacs = [(w, tac) for tac, w in res_dict.items()]
        common_tacs.sort()
        return parse_strategy(common_tacs[-1][1])

    def _calc_tac_weights(self, tac_name, data, data_mask, time_limit):
        values = [(data_name, weight)
                  for data_name in data if data_name not in data_mask
                  for tac, weight in self.item.get(data_name, []) if tac == tac_name and weight < time_limit]
        data_names, weights = [], []
        if len(values) > 0:
            data_names, weights = tuple(zip(*values))
        return sum(weights) / len(weights) if len(weights) else time_limit, tac_name, data_names

    def get_par_tacs(self, data: list[str], time_limit: list) -> tuple[list[objects.Tactic], int]:
        assert not self.simple
        num_select = len(time_limit)
        if len(data) == 0:
            return [objects.Tactic('skip') for _ in range(num_select)], 0
        strategy2choose = []
        data_mask = set()
        data = [data_name for data_name in data if len(self.item.get(data_name, [])) > 0]
        while len(strategy2choose) < num_select and len(data_mask) < len(data):
            strategy_i = len(strategy2choose)
            strategy_set = set([tac for data_name in data for tac, _ in self.item.get(data_name, [])
                                if data_name not in data_mask])
            strategy_values = [self._calc_tac_weights(strategy, data, data_mask, time_limit[strategy_i]) for strategy in
                               strategy_set]
            strategy_values.sort(key=lambda x: (time_limit[strategy_i] + 1) * len(x[2]) - x[0])
            _, tac_name, data_names = strategy_values[-1]
            if len(data_names) == 0:
                break
            strategy2choose.append(parse_strategy(tac_name))
            data_mask |= set(data_names)
        return strategy2choose, len(data) - len(data_mask)


class CompleteSolveCache:
    def __init__(self, timeout, cache=None, probe_cache=None):
        if cache is None:
            cache = {}
        if probe_cache is None:
            probe_cache = {}

        self.cache: dict[str, list[tuple[float, str]]] = cache
        self.probe_cache = probe_cache
        self.timeout = timeout

    def reset(self):
        self.cache = {}
        self.probe_cache = {}

    def load(self, path: str, clip: Optional[int] = None):
        if not path.endswith('.cac') or not os.path.exists(path):
            return
        if clip is None:
            clip = 0

        with open(path, 'r+') as cache_file:
            for line in cache_file:
                line = line.strip()
                data_name, res_list = line.split('|')

                res_list = [(cost, tac_name) for cost, tac_name in eval(res_list) if cost < self.timeout]
                if 0 < clip < len(res_list): res_list = res_list[:clip]

                self.cache[data_name] = res_list

        probe_path = f'{path[:-4]}.cpc'
        if os.path.exists(probe_path):
            with open(probe_path, 'r') as f:
                self.probe_cache = eval(f.readline())
            return

        # elif probe_cache not exists, create it.
        for data_name, tac_list in tqdm(self.cache.items(), desc='re-probe'):
            probe_dicts = {tac_name: _create_probe_dict(data_name, strategy2list(parse_strategy(tac_name)), self.timeout)
                           for _, tac_name in tac_list}
            self.probe_cache[data_name] = probe_dicts
        self.save_all(path)

    def save_all(self, path):
        if not path.endswith('.cac'):
            return

        with open(path, 'w') as f:
            for data_name, res_list, in self.cache.items():
                f.write(f'{data_name}|{str(res_list)}')

        with open(f'{path[:-4]}.cpc', 'w') as f:
            f.write(str(self.probe_cache))

    def init_batch(self, data: list[str], tac_seqs: list[TacSeqData], cache_path: Optional[str] = None,
                   batch_size: int = 8, cache_size: int = 3):

        cache_path = _fill_cache_path(cache_path, 'cac')
        strategies = [make_strategy(*seq) for _, seq in tac_seqs]
        tac_names = [str(strategy) for strategy in strategies]
        solve_res = solve_batch_cross(data, strategies, self.timeout, batch_size, reverse=True)
        for data_name, res_list in tqdm(zip(data, solve_res), desc="probe"):
            if data_name not in self.cache:
                self.cache[data_name] = []
            cache_list: list = self.cache[data_name]
            for tac_name, rtime in zip(tac_names, res_list):
                if rtime > self.timeout: continue
                insort(cache_list, (rtime, tac_name))
                if len(cache_list) > cache_size:
                    cache_list.pop()
            if cache_list:
                probe_dicts = {tac_name:
                                   _create_probe_dict(data_name, strategy2list(parse_strategy(tac_name)), self.timeout)
                               for _, tac_name in cache_list}
                self.probe_cache[data_name] = probe_dicts
        self.save_all(cache_path)


    def init_batch2(self, data: list[str], tac_seqs: list[TacSeqData], cache_path: Optional[str] = None,
                    batch_size: int = 8, cache_size: int = 3):
        warnings.warn('deprecated', DeprecationWarning)

        cache_path = _fill_cache_path(cache_path, 'cac')
        if os.path.exists(cache_path):
            self.load(cache_path, clip=cache_size)
            return

        for data_name in tqdm(data, desc='cache'):
            if data_name not in self.cache:
                self.cache[data_name] = []
            cache_list: list = self.cache[data_name]
            min_cost = self.timeout

            for tac_i in range(0, len(tac_seqs), batch_size):
                now_seqs = tac_seqs[tac_i: min(tac_i + batch_size, len(tac_seqs))]
                now_seqs = [make_strategy(*seq) for _, seq in now_seqs]

                res_list, _ = solve_batch_use_runner(data_name, now_seqs, min_cost, batch_size)

                for rtime, now_strategy in zip(res_list, now_seqs):
                    if rtime > min_cost:
                        continue
                    insort(cache_list, (rtime, str(now_strategy)))
                    if len(cache_list) > cache_size:
                        cache_list.pop()
                    if len(cache_list) == cache_size:
                        min_cost = cache_list[-1][0]

            if cache_list:
                probe_dicts = {tac_name:
                                   _create_probe_dict(data_name, strategy2list(parse_strategy(tac_name)), self.timeout)
                               for _, tac_name in cache_list}
                self.probe_cache[data_name] = probe_dicts
            self.save_all(cache_path)

    def get_probes(self, data_name, prefix):
        depth = len(prefix)
        tac_seqs = self.get_tac_seq(data_name, prefix, to_str=True)
        if data_name not in self.probe_cache or len(tac_seqs) == 0:
            return None
        for tac_name, _ in tac_seqs:
            if tac_name in self.probe_cache[data_name] and depth in self.probe_cache[data_name][tac_name]:
                return self.probe_cache[data_name][tac_name][depth]
        return None

    def create_data_dict(self, data: list[str], prefix: list[str]) -> CompleteDataDict:
        res = {data_name: self.get_tac_seq(data_name, prefix, to_str=True, calc_weight=False) for data_name in data}
        return CompleteDataDict(res, prefix)

    def get_tac_seq(self, data_name, prefix: list[str], to_str=False, calc_weight=True) -> list[
        tuple[objects.Tactic | str, float]]:
        tac_tuples = self.cache.get(data_name, [])
        strategies = []
        rtimes = []
        total_rtime = 1e-10
        for rtime, tac_name in tac_tuples:
            strategy = parse_strategy(tac_name)
            tac_seq = strategy2list(strategy)
            tac_seq_str = [str(s) for s in tac_seq]
            if len(prefix) >= len(tac_seq_str) or tac_seq_str[:len(prefix)] != prefix:
                continue
            total_rtime += (self.timeout - rtime)
            strategy = make_strategy(*tac_seq[len(prefix):])
            strategies.append(str(strategy) if to_str else strategy)
            rtimes.append(rtime)
        if calc_weight:
            weights = [(self.timeout - rtime) / total_rtime for rtime in rtimes]
        else:
            weights = [rtime for rtime in rtimes]
        return list(zip(strategies, weights))
