import sys
import warnings

sys.path.append('../..')

import random
from multiprocessing import Pool

from tqdm import trange

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing
from torch_geometric.data import Batch, Data
import time
import copy
import heapq

from sirismt.tools.tokenizer import BowTokenizer
from sirismt.agent.replay import ListSampleBuffer
from sirismt.agent.models import ReluDNN, DQN, DQNWithoutGNN
from sirismt.tools.solver import solve_batch_use_runner
from sirismt.tools.strategy import StrategyEnumerator
from sirismt.tools.env import SMTEnv
from sirismt.combiner.tuner import refine_tac_group


def _init_graphs(smt_instances: list[str]):
    formula_features = {}
    for smt_instance in smt_instances:
        with np.load(smt_instance.replace(".smt2", ".npz")) as npz_file:
            nodes = torch.tensor(npz_file['nodes']).float()
            edge_sets = ['AST', 'Back-AST', 'Data']
            edges = torch.tensor(npz_file['edges']).long()
            edge_attr = torch.tensor(npz_file['edge_attr']).float()
        if "AST" not in edge_sets:
            edges = torch.stack((edges[0][edge_attr != 0], edges[1][edge_attr != 0]))
            edge_attr = edge_attr[edge_attr != 0]
        if "Back-AST" not in edge_sets:
            edges = torch.stack((edges[0][edge_attr != 1], edges[1][edge_attr != 1]))
            edge_attr = edge_attr[edge_attr != 1]
        if "Data" not in edge_sets:
            edges = torch.stack((edges[0][edge_attr != 2], edges[1][edge_attr != 2]))
            edge_attr = edge_attr[edge_attr != 2]

        graph_data = Data(x=nodes,
                  edge_index=edges,
                  edge_attr=edge_attr,
                  problemType=torch.tensor([0]),
                  batch=torch.zeros(nodes.size(0), dtype=torch.int64))
        formula_features[smt_instance] = graph_data
    return formula_features


def _init_bows(smt_instances: list[str]):
    formula_features = {}
    tokenizer = BowTokenizer()
    for smt_instance in smt_instances:
        formula_features[smt_instance] = tokenizer.tokenize(smt_instance)
    return formula_features


class Agent:
    def __init__(self, config: dict, enumerator: StrategyEnumerator, device='cpu'):
        self.config = config
        self.all_tactics = enumerator.all_tactics

        self.online_net = None
        self.target_net = None

        self.optimizer = None

        self.buf = ListSampleBuffer(1000, 8)
        self.formula_features = {}

        self.rand_num = self.config['rand_episode_cnt']

        self.gamma = 0.5
        self.trans_cnt = 20

        self.best_strategy = {}
        self.r_denominator = None
        self.device = torch.device(device)

        self.passed_sequence = []
        self.change_count = 0

    def create_net(self, in_num: int):
        if self.config['feature_type'] == 'graph':
            self.online_net = (DQN(in_num, len(self.all_tactics), gnn_numAttentionLayers=1, gnn_pool='mean')
                               .to(self.device))
            self.target_net = (DQN(in_num, len(self.all_tactics), gnn_numAttentionLayers=1, gnn_pool='mean')
                               .to(self.device))
        elif self.config['feature_type'] == 'bow':
            tokenizer = BowTokenizer()
            self.online_net = (DQNWithoutGNN(in_num, tokenizer.token_length(), len(self.all_tactics))
                               .to(self.device))
            self.target_net = (DQNWithoutGNN(in_num, tokenizer.token_length(), len(self.all_tactics))
                               .to(self.device))
        else:
            self.online_net = ReluDNN(in_num, len(self.all_tactics)).to(self.device)
            self.target_net = ReluDNN(in_num, len(self.all_tactics)).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=1e-4, amsgrad=True)

    def __get_batch_features(self, batch_instance):
        res = []
        for instance in batch_instance:
            res.append(self.formula_features.get(instance))
        if self.config['feature_type'] == 'graph':
            return Batch.from_data_list(res).to(self.device)
        if self.config['feature_type'] == 'bow':
            return torch.as_tensor(res).float().to(self.device)
        return []

    def do_train(self):
        try:
            batch_s, batch_a, batch_r, batch_res, batch_s_, batch_instance = self.buf.sample()
            if batch_s is None:
                return
            tensor_r = torch.as_tensor([batch_r]).to(self.device)
            tensor_a = torch.as_tensor([batch_a]).to(self.device)
            tensor_s = torch.as_tensor(batch_s).to(self.device)
            tensor_s_ = torch.as_tensor(batch_s_).to(self.device)
            feature_batch = self.__get_batch_features(batch_instance)

            next_q_values = self.online_net(tensor_s_, feature_batch)
            target_q_values = self.target_net(tensor_s_, feature_batch)

            target_max_values = target_q_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))

            targets = torch.transpose(tensor_r, 1, 0) + self.gamma * target_max_values
            targets = torch.transpose(targets, 0, 1)

            action_value = self.online_net(tensor_s, feature_batch).gather(dim=1, index=tensor_a)
            criterion = nn.SmoothL1Loss()
            loss = criterion(action_value, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except torch.cuda.OutOfMemoryError:
            self.buf.BATCH_SIZE = max(self.buf.BATCH_SIZE//2, 2)
        self.buf.BATCH_SIZE = min(self.buf.BATCH_SIZE+1, 32)

    def construct_denominator(self, smt_instances: list):
        """use z3 solver to construct denominator which can let reward make sense"""
        res_list, _ = solve_batch_use_runner(smt_instances, None, batch_size=8)
        avg_r = sum(res_list) / len(smt_instances)
        self.r_denominator = avg_r / 10

    def choose_tactic_by_weight(self, act_weights, lst_tactic=None) -> int:
        if lst_tactic is None:
            lst_tactic = []
        if act_weights is None:
            ind = random.randint(0, len(self.all_tactics)-1)
            while ind in lst_tactic:
                ind = random.randint(0, len(self.all_tactics)-1)
            return ind

        act_weights = act_weights.sub(torch.min(act_weights)).add_(0.5)
        torch.manual_seed(int(random.random() * 100000))
        act_weights = nn.functional.normalize(act_weights, dim=0)
        while act_weights[torch.argmin(act_weights).item()] < 0:
            act_weights[torch.argmin(act_weights).item()] = 0.5
        ind = torch.multinomial(act_weights, 1).item()
        while ind in lst_tactic and act_weights[torch.argmax(act_weights).item()] > 0:
            act_weights[torch.argmax(act_weights).item()] = -10000
            ind = int(torch.argmax(act_weights).item())
        return ind

    def choose_tactic_within_train(self, episode: int, state: torch.Tensor, feature, lst_tactic: list = None) -> int:
        """choose a tactic within train, return it by its str name"""
        if lst_tactic is None:
            lst_tactic = []
        if episode < self.rand_num:
            return self.choose_tactic_by_weight(None)
        with torch.no_grad():
            weights = self.online_net(torch.as_tensor(state).to(self.device),
                                      feature)[0]
        return self.choose_tactic_by_weight(weights, lst_tactic)

    def featurize_instance(self, instance: str):
        """featurize needed property from instance, which may use in further study."""
        if self.config['feature_type'] == 'graph':
            return Batch.from_data_list([self.formula_features.get(instance)]).to(self.device)
        if self.config['feature_type'] == 'bow':
            return torch.as_tensor(self.formula_features.get(instance)).float().to(self.device)
        return {}

    def calc_reward(self, rtime, len_passed_seq, solved=False):
        bias = 10 - len_passed_seq
        bias += 0 if not solved else 20
        reward = bias - rtime / self.r_denominator
        return reward

    def _apply_tactic(self, queue, env, instance, ind):
        if isinstance(ind, str):
            ind = self.all_tactics.index(ind)
        s, s_, solved, rtime = env.step(ind)

        if rtime < 0:
            queue.put(
                ('buf', ([s, ind, -10.0, solved, s.copy(), instance], solved))
            )
            return False, -1

        bias = 20 - len(env.tried_action)
        bias += 0 if solved else 20
        reward = self.calc_reward(rtime, len(env.tried_action), solved)
        queue.put(
            ('buf', ([s, ind, reward, solved, s_, instance], solved))
        )
        return solved, rtime

    def train_with_one_formula(self, queue, step_cnt: int, instance: str, episode: int):
        """train with one formula and return the solve reward and tactic sequence within train"""
        features = self.featurize_instance(instance)
        episode_reward = 0
        env = SMTEnv(self.all_tactics)
        env.reset(instance)
        s = env.observe_state()
        solved = False

        for st_i in range(step_cnt):
            if env.is_full_try():
                break
            tensor_s = torch.as_tensor(s).unsqueeze(0).to(self.device)
            ind = self.choose_tactic_within_train(episode, tensor_s, features, env.tried_action)
            solved, rtime = self._apply_tactic(queue, env, instance, ind)
            episode_reward -= max(rtime, 0)
            if solved:
                break

        res_reward, res_seq = episode_reward, env.get_tac_seq()
        tac_seq = res_seq
        if not solved:
            res_tuple = ('res', (instance, -1000, [], []))
            queue.put(
                res_tuple
            )
            return
        passed_sequence, origin_length = self.passed_sequence, len(self.passed_sequence)
        passed_sequence.append(tac_seq)
        episode_reward = res_reward

        queue.put(
            ('res', (instance, episode_reward, res_seq, passed_sequence[origin_length:]))
        )

        for i in range(2):
            opt_seqs = refine_tac_group([instance], [tac_seq], passed_sequence)
            for opt_seq in opt_seqs:
                seq_reward = self.rebuild_seq(queue, instance, opt_seq)
                if seq_reward > 0:
                    passed_sequence.append(opt_seq)
                if seq_reward > episode_reward:
                    res_reward, res_seq = seq_reward, opt_seq
                    queue.put(
                        ('res', (instance, episode_reward, res_seq, passed_sequence[origin_length:]))
                    )
        return

    def rebuild_seq(self, queue, instance, tac_seq):
        total_reward = 0
        env = SMTEnv(self.all_tactics)
        env.reset(instance)
        for act in tac_seq:
            if act == 'skip':
                continue
            ind = self.all_tactics.index(act)
            solved, rtime = self._apply_tactic(queue, env, instance, ind)
            if rtime < 0:
                return -10000
            total_reward -= rtime
            if solved:
                return total_reward
        return -10000

    def init_before_train(self, smt_instances: list):
        if self.r_denominator is None or self.r_denominator <= 0:
            # self.construct_denominator(smt_instances)
            self.r_denominator = 10
        if self.target_net is not None:
            self.target_net.load_state_dict(self.online_net.state_dict())
        for instance in smt_instances:
            self.best_strategy[instance] = []
        env = SMTEnv(self.all_tactics)
        env.reset(smt_instances[0])
        s = env.observe_state()
        self.create_net(len(s))

        if self.config['feature_type'] == 'graph':
            self.formula_features = _init_graphs(smt_instances)
        elif self.config['feature_type'] == 'bow':
            self.formula_features = _init_bows(smt_instances)

    def _collect_queue_msg(self, queue):
        while not queue.empty():
            msg, item = queue.get()
            if msg == 'buf':
                sample, done = item
                self.buf.add_sample(sample, done=done)
                if len(self.buf) > 1:
                    self.do_train()
            if msg == 'res':
                instance, reward, tac_seq, passed_sequence = item
                self.passed_sequence.extend(passed_sequence)
                if len(self.passed_sequence) > 500:
                    self.passed_sequence = self.passed_sequence[-500:]
                if tac_seq:
                    if (len(self.best_strategy) > 0 and
                            any([seq == tac_seq for _, _, seq in self.best_strategy[instance]])):
                        continue
                    if len(self.best_strategy[instance]) < 2:
                        heapq.heappush(self.best_strategy[instance], (reward, self.change_count, tac_seq))
                    else:
                        heapq.heappushpop(self.best_strategy[instance], (reward, self.change_count, tac_seq))
                    self.change_count += 1
                break


    def train_episode(self, queue, instances, ep_cnt, ep_i, batch_size, time_limit: float = None):
        change_cnt = 0

        with Pool(processes=batch_size) as pool:
            async_results = []
            for i, instance in enumerate(instances):
                async_result = pool.apply_async(
                    self.train_with_one_formula,
                    (queue, ep_cnt, instance, ep_i)
                )
                async_results.append(async_result)

            last_time = time.monotonic()

            with trange(len(instances), desc=f'ep:{ep_i}: ') as pbar:
                while len(async_results) > 0:
                    dead = []
                    for result in async_results:
                        if result.ready():
                            dead.append(result)
                        if time.monotonic() - last_time > time_limit:
                            dead.append(result)
                            print("\ntimeout process\n")
                    for result in dead:
                        async_results.remove(result)
                        change_cnt += 1
                        pbar.update(1)
                    if change_cnt > self.trans_cnt:
                        change_cnt -= self.trans_cnt
                        self.target_net.load_state_dict(self.online_net.state_dict())
                        print(f'ep:{ep_i} change model success')
                    self._collect_queue_msg(queue)
                    time.sleep(0.1)
                    if dead: last_time = time.monotonic()
        self._collect_queue_msg(queue)
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_episode2(self, queue, instances, ep_cnt, ep_i, batch_size, time_limit: float = None):
        warnings.warn("deprecated, but stable than train_episode", DeprecationWarning)
        batch_ids = list(range(batch_size))
        processes = []
        now_index = 0
        change_cnt = 0
        true_ep = ep_i
        ep_i = -50 if ep_i == 0 else 0

        for _ in trange(len(instances), desc=f'ep:{true_ep}: '):
            while len(batch_ids) > 0 and now_index < len(instances):
                batch_id = batch_ids.pop()
                p = multiprocessing.Process(target=self.train_with_one_formula,
                                            args=(queue, ep_cnt, instances[now_index], ep_i,))
                p.start()
                t_before = time.monotonic()
                processes.append((batch_id, t_before, p))
                now_index += 1
                if ep_i < 1: ep_i += 1
            while processes:
                t_after = time.monotonic()
                dead_index = -1
                for index, p_tuple in enumerate(processes):
                    batch_id, t_before, p = p_tuple
                    p.join(0.1)
                    if not p.is_alive() or t_after-t_before > time_limit:
                        dead_index = index
                        break
                self._collect_queue_msg(queue)
                if dead_index < 0:
                    continue
                batch_id, _, p = processes.pop(dead_index)
                p.terminate()
                batch_ids.append(batch_id)
                change_cnt += 1
                break
            if change_cnt > self.trans_cnt:
                change_cnt -= self.trans_cnt
                self.target_net.load_state_dict(self.online_net.state_dict())
                print(f'ep:{ep_i} change model success')

    def train(self, smt_instances: list, episode_cnt: int = None, batch_size=8):
        """use the dataset(smt_instances) to train episode_cnt's turn"""
        #TODO extract timeout
        self.init_before_train(smt_instances)
        if episode_cnt is None:
            episode_cnt = self.config['episode_cnt']

        multiprocessing.set_start_method('spawn')
        multiprocessing.set_sharing_strategy('file_system')
        queue = multiprocessing.Manager().Queue(-1)
        ep_cnt = self.config['step_cnt']

        for ep_i in range(episode_cnt):
            last_best_strategies = copy.deepcopy(self.best_strategy)
            self.train_episode(queue, smt_instances, ep_cnt, ep_i, batch_size, 400)

            diff_instances = [instance for instance in smt_instances
                              if self.best_strategy[instance] != last_best_strategies[instance]]

            if len(diff_instances) == 0:
                print("stop caused by early-stop")
                break

            added_instance = [instance for instance in smt_instances if len(self.best_strategy[instance]) == 0]
            smt_instances = added_instance + diff_instances

            self.output_best_strategy()
            if self.config['exp_name'] is None:
                continue
            torch.save(self.online_net.state_dict(),
                       f'cache/model/{self.config["exp_name"]}_{ep_i}_5.pth')

    def output_best_strategy(self):
        """output the best tactic sequences found during train to out_file"""
        best_strategy = self.best_strategy
        if self.config['out_file'] is None:
            return
        with open(self.config['out_file'], 'w') as f:
            for instance, tac_seq in best_strategy.items():
                f.write(f'{instance}->{str(tac_seq)}\n')
