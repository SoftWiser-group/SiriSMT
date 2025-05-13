import sys
sys.path.append('../..')

import copy
import math
import random
from sirismt.tools.solver import solve_batch_use_runner
from sirismt.tools import objects

nxt_id = 0


class BenchmarkSeqReward:
    def __init__(self, tac_seq, instances):
        global nxt_id
        self.benchmark = nxt_id
        nxt_id += 1
        self.seq = tac_seq
        if len(tac_seq) > 0:
            self.tactic = objects.AndThen(*tac_seq) if len(tac_seq) > 1 else tac_seq[0]
            if isinstance(self.tactic, str):
                self.tactic = objects.Tactic(self.tactic)
        else:
            self.tactic = objects.Tactic('skip')
        self.instances = instances
        self.reward = 0.0
        self.eval()

    def eval(self):
        eval_res, _ = solve_batch_use_runner(self.instances, self.tactic, batch_size=1)
        self.reward = -sum(eval_res)

    def trans_to(self, x):
        sub_seq = [e for x_i, e in zip(x, self.seq) if x_i == 1]
        return BenchmarkSeqReward(sub_seq, self.instances)


def get_bsr(bsr: BenchmarkSeqReward, x) -> BenchmarkSeqReward:
    return bsr.trans_to(x)


def seq2str(bsr):
    return str([str(t) for t in bsr.seq])


def bsr1_better_bsr2(new_bsr, bsr):
    if new_bsr.reward > bsr.reward:
        return True
    return False


def phi(bsr: BenchmarkSeqReward, x):
    new_bsr = get_bsr(bsr, x)
    if bsr1_better_bsr2(new_bsr, bsr):
        return True, new_bsr
    else:
        return False, None


def max_benchmark_seq_reward(final_bsr, new_bsr):
    if bsr1_better_bsr2(final_bsr, new_bsr):
        return final_bsr
    else:
        return new_bsr


def prob_dd(bsr: BenchmarkSeqReward, prob: []):
    final_bsr = copy.deepcopy(bsr)
    n = len(bsr.seq)

    xt = [1] * n
    while True:
        if all(math.isclose(value, 1) or math.isclose(value, 0) for value in prob):
            return final_bsr

        choice = [(prob[i], i) for i in range(n) if xt[i] == 1]
        x = xt.copy()
        assert x == xt
        choice = sorted(choice)
        m = len(choice)

        expect, pass_pro = 0.0, 1.0
        for i in range(m):
            p, idx = choice[i]
            x[idx] = 0
            pass_pro *= 1 - p
            new_expect = (i + 1) * pass_pro
            if expect > new_expect:
                x[idx] = 1
                break
            else:
                expect = new_expect

        # update probability
        t_or_f, new_bsr = phi(bsr, x)
        if t_or_f:
            final_bsr = max_benchmark_seq_reward(final_bsr, new_bsr)
            prob = [0 if x[i] == 0 else prob[i] for i in range(n)]
            xt = x.copy()
        else:
            tmp = 1
            for i in range(n):
                if x[i] == 0:
                    tmp *= 1-prob[i]
            # math.prod(1 - prob[i] for i in range(n) if x[i] == 0)
            prob = [prob[i] / (1 - tmp) if x[i] == 0 else prob[i] for i in range(n)]


def iterative_prob_dd(tac_seq, instances, iter_times=1) -> []:
    bsr = BenchmarkSeqReward(tac_seq, instances)
    new_bsr = copy.deepcopy(bsr)
    for it in range(iter_times):
        random_list = [random.uniform(0.3, 0.7) for _ in range(len(bsr.seq))]
        temp_bsr = prob_dd(new_bsr, random_list)
        new_bsr = max_benchmark_seq_reward(new_bsr, temp_bsr)
    return new_bsr.seq
