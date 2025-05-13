from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import subprocess
import tempfile
import threading
import time

import z3
from tqdm import tqdm

import sys

sys.path.append('../..')

from sirismt.tools.types import *

__all__ = [
    'Z3Runner',
    'solve_with_timeout',
    'solve_use_runner',
    'solve_batch_cross',
    'solve_batch_use_runner',
    'get_probes'
]


class Z3Runner(threading.Thread):
    def __init__(self, smt_file: str, timeout: int, strategy: Optional[str] = None) -> None:
        super().__init__()
        self.smt_file = smt_file
        self.timeout = timeout
        self.strategy = strategy
        self.p: Optional[subprocess.Popen] = None

        self.time_before = self.time_after = 0
        self.new_file_name: str = smt_file

        if self.strategy is None:
            return

        os.makedirs('tmp', exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile('w', suffix='.smt2', dir='tmp', delete=False)
        self.new_file_name = temp_file.name

        with open(smt_file, 'r') as f:
            for line in f:
                new_line = '(check-sat-using %s)\n' % strategy if 'check-sat' in line else line
                temp_file.write(new_line)
        temp_file.close()

    def run(self) -> None:
        self.time_before = time.monotonic()
        z3_cmd = ['z3', '-smt2', self.new_file_name, '-st', f'-T:{self.timeout}']

        try:
            self.p = subprocess.Popen(z3_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.time_before = time.monotonic()
            self.p.wait()
        except Exception:
            pass
        finally:
            self.time_after = time.monotonic()

    def collect(self) -> tuple[str, int, float] | tuple[None, None, None]:
        if self.is_alive():
            try:
                if self.p:
                    self.p.terminate()
                self.join()
            except Exception:
                pass
            self._clean_temp_file()
            return None, None, None
        elif not self.p:
            self._clean_temp_file()
            return None, None, None

        out, err = self.p.communicate()
        self._clean_temp_file()

        res = None
        out = out.decode('utf-8').strip()
        lines = out.splitlines() if out else []

        if lines and lines[0] in {'sat', 'unsat', 'unknown'}:
            res = lines[0]

        rlimit = None
        for line in lines:
            if 'rlimit' not in line: continue

            tokens = line.split(' ')
            for token in tokens:
                if token.isdigit():
                    rlimit = int(token)
                    break

        elapsed = self.time_after - self.time_before
        if elapsed < 0: elapsed = None

        return res, rlimit, elapsed

    def _clean_temp_file(self) -> None:
        if self.strategy is None or not os.path.exists(self.new_file_name):
            return

        try:
            os.remove(self.new_file_name)
        except OSError:
            pass


def get_probes(formula: z3.AstVector, allowed_probes: Optional[list[str]] = None) -> list[float]:
    if allowed_probes is None:
        allowed_probes = z3.probes()
    probes = [z3.Probe(p) for p in allowed_probes]
    goal = z3.Goal()
    goal.add(formula)
    return [probe(goal) for probe in probes]


def solve_use_runner(smt_instance: str, tactic: str, timeout=10) -> tuple[str, int, float]:
    runner = Z3Runner(smt_instance, timeout, tactic)
    runner.start()
    runner.join(timeout)
    res, rlimit, rtime = runner.collect()
    return str(res), rlimit, rtime


def solve_with_timeout(instance: str, tactic: Strategy, timeout: float = 10) \
        -> tuple[str, float, z3.AstVector] | tuple[str, None, None]:
    formula = z3.parse_smt2_file(instance)
    solver = tactic.tactic.solver()
    solver.set('timeout', timeout * 1000)
    solver.add(formula)
    t_before = time.monotonic()
    res = str(solver.check())
    t_after = time.monotonic()
    if t_after - t_before > timeout or res not in {'sat', 'unsat', 'unknown'}:
        return 'unknown', None, None
    return str(res), t_after - t_before, solver.assertions()


def _create_runner(smt_instance: str, tactic: str | Strategy, timeout: int) -> Z3Runner:
    if isinstance(tactic, str) or tactic is None:
        runner = Z3Runner(smt_instance, timeout, tactic)
    else:
        runner = Z3Runner(smt_instance, timeout, tactic.to_smt2() if tactic is not None else None)
    return runner


def solve_batch_use_runner(smt_instance: str | list[str],
                           tactic: Strategy | list[Strategy] | list[str] = None,
                           timeout: int | list = 10,
                           batch_size: int = 4,
                           time_cost: float = None) -> tuple[list[float], list[str]]:
    if time_cost is None:
        time_cost = 2 * timeout
    assert isinstance(smt_instance, list) or isinstance(tactic, list)
    if not isinstance(smt_instance, list):
        smt_instance = [smt_instance for _ in range(len(tactic))]
    if not isinstance(tactic, list):
        tactic = [tactic for _ in range(len(smt_instance))]
    if not isinstance(timeout, list):
        timeout = [timeout for _ in range(len(smt_instance))]

    res_list = []
    runner_list = []
    failed_list = []

    def clear_runner_list():
        for job in runner_list:
            now_time = time.monotonic()
            job.join(max(0.0, job.time_before + job.timeout - now_time))
            res, rlimit, rtime = job.collect()
            if res is None or res == 'unknown' or rtime is None or rtime > job.timeout:
                res_list.append(time_cost)
                failed_list.append(job.smt_file)
            else:
                res_list.append(rtime)
        runner_list.clear()

    for instance_, tactic_, timeout_ in zip(smt_instance, tactic, timeout):
        runner = _create_runner(instance_, tactic_, timeout_)
        runner.start()
        runner_list.append(runner)
        if len(runner_list) % batch_size == 0:
            clear_runner_list()
    clear_runner_list()
    return res_list, failed_list


def _solve_use_runner_wrapper(smt_instance: str, tactic: str, timeout=10):
    res = solve_use_runner(smt_instance, tactic, timeout=timeout)
    return (smt_instance, tactic), res


def solve_batch_cross(instances: str | list[str],
                      tactics: Strategy | list[Strategy] | list[str] = None,
                      timeout: int | list = 10,
                      batch_size: int = 4,
                      time_cost: float = None,
                      reverse: bool = False) -> list[list[float]]:
    # TODO: add cache by solid|hash file
    if time_cost is None:
        time_cost = 2 * timeout
    if not reverse:
        res_list = [[time_cost for _ in instances] for _ in tactics]
    else:
        res_list = [[time_cost for _ in tactics] for _ in instances]

    tactics = [tactic.to_smt2() if tactic is not None else None for tactic in tactics]
    tactic_index = {tactic: tac_i for tac_i, tactic in enumerate(tactics)}
    instance_index = {instance: ins_i for ins_i, instance in enumerate(instances)}

    with ProcessPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(_solve_use_runner_wrapper, instance, tactic, timeout)
                   for instance in instances
                   for tactic in tactics]

        with tqdm(desc='cross', total=len(futures)) as pbar:
            for future in as_completed(futures):
                params, res = future.result()
                instance, tactic = params
                res, _, rtime = res
                if res is None or res == 'unknown' or rtime is None or rtime > timeout:
                    continue
                if not reverse:
                    res_list[tactic_index[tactic]][instance_index[instance]] = rtime
                else:
                    res_list[instance_index[instance]][tactic_index[tactic]] = rtime
                pbar.update(1)

    return res_list

