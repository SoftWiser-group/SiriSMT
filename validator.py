import argparse
import os.path
import shlex
import subprocess
import threading
import time

import abc
import bisect

from sirismt.tools.solver import Z3Runner

class AbstractTestMaker(metaclass=abc.ABCMeta):
    def __init__(self, name:str) -> None:
        self.name = name

    @abc.abstractmethod
    def make_test(self, data: str, timeout: int) -> Z3Runner:
        pass

class CVC5Runner(Z3Runner):
    def __init__(self, smt_file: str, timeout: int):
        super().__init__(smt_file, timeout)

    def run(self) -> None:
        self.time_before = time.monotonic()
        cvc5_cmd = f'cvc5 --stats {self.new_file_name}'

        self.p = subprocess.Popen(shlex.split(cvc5_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.p.wait()

    def collect(self) -> tuple[str, int, float] | tuple[str, None, None]:
        if self.is_alive():
            try:
                self.p.terminate()
                self.join()
            except OSError:
                pass
            return 'unknown', None, None
        out, err = self.p.communicate()

        lines = out[:-1].decode("utf-8").split('\n')
        res = 'unknown'
        if lines[0] in ['sat', 'unsat']: res = lines[0]
        lines = err[:-1].decode("utf-8").split('\n')
        timeout = None
        for line in lines:
            if line.startswith('global::totalTime = '):
                timeout = eval(line[20:-2]) / 1000
        return res, 0, timeout


class StrategyTestMaker(AbstractTestMaker):
    def __init__(self, name: str, smt2_strategy: str = None):
        super().__init__(name)
        self.strategy = smt2_strategy

    def make_test(self, data: str, timeout: int) -> Z3Runner:
        return Z3Runner(data, timeout, self.strategy)


class WithoutTheoryTestMaker(AbstractTestMaker):
    class BasicZ3Runner(Z3Runner):
        def __init__(self, smt_file: str, timeout: int, file_id: int = 1):
            threading.Thread.__init__(self)
            self.smt_file = smt_file
            self.timeout = timeout
            self.p = None
            self.time_before = self.time_after = 0

            self.tmp_file = open('strategytrainner/tmp/tmp_valid_{}.smt2'.format(file_id), 'w')
            with open(self.smt_file, 'r') as f:
                for line in f:
                    new_line = line
                    if 'set-logic' in line:
                        continue
                    self.tmp_file.write(new_line)
            self.tmp_file.close()
            self.new_file_name = 'strategytrainner/tmp/tmp_valid_{}.smt2'.format(file_id)
    def __init__(self):
        super().__init__("z3-basic")

    def make_test(self, data: str, timeout: int) -> Z3Runner:
        return WithoutTheoryTestMaker.BasicZ3Runner(data, timeout)

class CVC5TestMaker(AbstractTestMaker):
    def __init__(self):
        super().__init__('cvc5')

    def make_test(self, data: str, timeout: int) -> Z3Runner:
        return CVC5Runner(data, timeout)


class TestManager:
    def __init__(self, main_strategy: AbstractTestMaker, batch_size: int, timeout: int, file_id_base: int):
        self.solved = [0, [0], [0], 0, [0]] # all, only, can, none, best
        self.time_cost = []
        self.par2_cost = [0]

        self.batch_size = batch_size
        self.timeout = timeout

        self.file_id_base = file_id_base
        self.file_id_offset = 0
        self.offset_max = 50

        self.strategies = [main_strategy]
        self.total_passed = 0
        self.pattern = main_strategy.name

    def register(self, strategy: AbstractTestMaker) -> None:
        for i in [1, 2, 4]:
            self.solved[i].append(0)
        self.time_cost.append([])
        self.par2_cost.append(0)
        self.strategies.append(strategy)
        self.pattern = ", ".join([strategy.name for strategy in self.strategies])

    def next_test_case(self, test_data):
        test_batch = []
        for test_case in test_data:
            test_tuple = [maker.make_test(test_case, self.timeout) for maker in self.strategies]
            for e in test_tuple:
                e.start()
            test_batch.append(test_tuple)
            if len(test_batch) >= self.batch_size:
                for case in test_batch:
                    yield case
                test_batch.clear()
        for test_case in test_batch:
            yield test_case

    def collect_test_result(self, test_case):
        self.total_passed += 1
        solved_num = 0
        test_res = []
        best_rtime = self.timeout * 3

        for test_i, test in enumerate(test_case):
            now_time = time.monotonic()
            test.join(max(0, test.time_before + self.timeout - now_time))
            res, _, rtime = test.collect()

            if res not in ["sat", "unsat"] or rtime is None or rtime > self.timeout:
                res = None
                rtime = self.timeout * 2
            best_rtime = min(best_rtime, rtime)
            solved_num += res is not None
            test_res.append(rtime if res is not None else None)

        if solved_num == 3:
            self.solved[0] += 1
        elif solved_num == 0:
            self.solved[3] += 1

        for res_i, res in enumerate(test_res):
            if res is None:
                self.par2_cost[res_i] += self.timeout * 2
                continue
            self.par2_cost[res_i] += res
            self.solved[2][res_i] += 1
            if solved_num == 1:
                self.solved[1][res_i] += 1
            if res == best_rtime:
                self.solved[4][res_i] += 1
            if res_i > 0 and test_res[0] is not None:
                self.time_cost[res_i - 1].append(res / test_res[0])
        self.print()

    def run(self, test_data):
        for test_case in self.next_test_case(test_data):
            self.collect_test_result(test_case)

    def print_ratio(self):
        ratio_points = [0.1, 0.5, 0.9]
        for strategy, ratio_list in zip(self.strategies[1:], self.time_cost):
            if len(ratio_list) == 0:
                continue
            ratio_list.sort()
            print(f"====print time speed up with {strategy.name}")
            for ratio_point in ratio_points:
                print(f"{ratio_point}x speed up: {ratio_list[int(ratio_point*len(ratio_list))]}")
            print(f"avg speed up: {sum(ratio_list) / len(ratio_list)}")
            print(f"1x point: {bisect.bisect_left(ratio_list, 1.0)/len(ratio_list)}")

    def print(self):
        print(f"=======for total {self.total_passed} test data:")
        print(f"all  solved: {self.solved[0]}")
        print(f"none solved: {self.solved[3]}")
        print(f"only [{self.pattern}]: {self.solved[1]}")
        print(f"can  [{self.pattern}]: {self.solved[2]}")

        self.print_ratio()

        print("===print par2 time:")
        for strategy, par2 in zip(self.strategies, self.par2_cost):
            print(f"{strategy.name:<8}: {par2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_strategy', type=str)
    parser.add_argument('--base_strategy', type=str, default=None)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--timeout', type=int)
    parser.add_argument("--file_id", type=int, default=10000)
    args = parser.parse_args()
    test_data = []
    for root, paths, files in os.walk(args.test_data):
        for filename in files:
            if str(filename).endswith(".smt2"):
                test_data.append(os.path.join(root, filename))

    with open(args.main_strategy, 'r') as f:
        main_strategy = f.readline()
    manager = TestManager(StrategyTestMaker('main', main_strategy), args.batch_size, args.timeout, args.file_id)

    if args.base_strategy is not None:
        with open(args.base_strategy, 'r') as f:
            base_strategy = f.readline()
        manager.register(StrategyTestMaker('base', base_strategy))

    manager.register(StrategyTestMaker('z3-logic', None))
    manager.register(WithoutTheoryTestMaker())

    manager.run(test_data)


if __name__ == '__main__':
    main()
