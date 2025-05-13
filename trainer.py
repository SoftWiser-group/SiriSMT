import random
import time

import numpy.random
import torch
import torch_geometric

from sirismt.agent.agent import Agent
from sirismt.combiner.tuner import RandomTuner
from sirismt.combiner.combiner import C45Combiner
from sirismt.tools.strategy import StrategyEnumerator
from sirismt.tools.transformer import *

import argparse
import os
import logging
import datetime
import yaml


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


def load_folder(folder_name: str) -> list:
    if folder_name is None:
        return []
    content = []
    for root, paths, files in os.walk(folder_name):
        for filename in files:
            if filename.endswith("smt2"):
                content.append(os.path.join(root, filename))
    return content


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)
    pass


class Trainer:
    def __init__(self, configs: dict, timeout, train_data: list = None, valid_data: list = None, device: str = 'cpu'):

        self.logger = logging.getLogger('Trainer')

        self.tactics_config = configs['tactics_config']
        self.agent_config = configs['agent_config']
        self.tuner_config = configs['tuner_config']
        self.combiner_config = configs['combiner_config']

        enumerator = StrategyEnumerator(**self.tactics_config)

        self.agent = Agent(self.agent_config, enumerator, device)

        self.tuner = RandomTuner(self.tuner_config, enumerator)

        self.combiner = C45Combiner(timeout)

        self.train_data = train_data
        self.valid_data = valid_data

        self.candidate_tac_seq = None
        self.tuned_tac_seq = None
        self.final_strategy = None


    def load_candidate_tactic_sequence(self, path: str):
        """load candidate tactic sequence stored in file, it can skip the train step"""
        self.candidate_tac_seq = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '':
                    continue
                instance, tac_seqs = line.split('->')
                tac_seqs = eval(tac_seqs)
                if len(tac_seqs) == 0:
                    continue
                for _, _, tac_seq in tac_seqs:
                    if len(tac_seq) == 0:
                        continue
                    if tac_seq[0] == 'skip':
                        tac_seq = tac_seq[1:]
                    self.candidate_tac_seq.append((instance, tac_seq))
        self.logger.info('load {} candidate tactic sequences completely'.format(len(self.candidate_tac_seq)))

    def train_candidate_tactic_sequence(self, batch_size=8):
        """
        use RL to train a model predict which tactic should be used to solve formula,
        collect a set of candidate tactic sequences
        """
        self.candidate_tac_seq = None
        t_before = time.monotonic()
        self.agent.train(self.train_data, batch_size=batch_size)
        best_tac_seqs = self.agent.best_strategy
        t_after = time.monotonic()
        self.logger.info(f'stage train wall-time: {t_after-t_before}')
        self.candidate_tac_seq = []
        for instance, tac_seqs in best_tac_seqs.items():
            for _, _, tac_seq in tac_seqs:
                if len(tac_seq) < 1:
                    continue
                if tac_seq[0] == 'skip':
                    tac_seq = tac_seq[1:]
                self.candidate_tac_seq.append((instance, tac_seq))

    def load_tuned_candidate_seq(self, path):
        """load tuned candidate sequence in file, it can skip the tuner step"""
        self.tuned_tac_seq = []
        with open(path, 'r') as f:
            for line in f.readlines():
                self.tuned_tac_seq.append(parse_strategy(line))
        self.logger.info('load {} tuned tactic sequences completely'.format(len(self.tuned_tac_seq)))

    def tuner_candidate_seq(self, batch_size=8):
        """tune the candidate sequences, store the best of them"""
        if self.candidate_tac_seq is None:
            self.train_candidate_tactic_sequence()
        t_before = time.monotonic()
        self.tuned_tac_seq = self.tuner.tuning(self.candidate_tac_seq, batch_size=batch_size)
        t_after = time.monotonic()
        self.logger.info(f'stage refine wall-time: {t_after-t_before}')

    def make_strategy(self, batch_size=8):
        """use tuned candidate sequences to make a strategy"""
        if self.tuned_tac_seq is None:
            self.tuner_candidate_seq()
        if len(self.valid_data) < 200:
            self.valid_data += self.train_data
        else:
            self.valid_data = [self.valid_data[data_i]
                               for data_i in range(0, len(self.valid_data), int(len(self.valid_data) / 200))]
        t_before = time.monotonic()
        tac_seqs = [strategy2list(st) for st in self.tuned_tac_seq]
        self.final_strategy = self.combiner.gen_strategy(self.valid_data, tac_seqs,
                                                         self.combiner_config['cache_path'],
                                                         batch_size=batch_size
                                                         )
        t_after = time.monotonic()
        self.logger.info(f'stage synthesis wall-time: {t_after-t_before}')
        if self.combiner_config['out_file'] == '':
            return
        with open(self.combiner_config['out_file'] + str(datetime.datetime.now()).replace(' ', ''), 'w') as f:
            f.write(self.final_strategy.to_smt2())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='experiments/configs/normal_config.yml',
                        help='path of configuration file like configs/normal_config.json')
    parser.add_argument('--train_data', type=str, default=None,
                        help='folder path of training data')
    parser.add_argument('--valid_data', type=str, default=None,
                        help='folder path of valid data, used in combiner.')
    parser.add_argument('--seed', type=int, default=None,
                        help='control the initial seed for program.')
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--without_integration', action='store_true')
    args = parser.parse_args()

    assert args.batch_size > 0

    if args.seed is None:
        args.seed = random.randint(0, int(1e9))
    print('use seed {0}'.format(args.seed))
    setup_seed(args.seed)

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    train_data = load_folder(args.train_data)
    valid_data = load_folder(args.valid_data)

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    trainer = Trainer(configs, args.timeout, train_data=train_data, valid_data=valid_data, device=device)

    load_config = configs['load_config']
    if load_config['candidate_tac_path'] != '':
        trainer.load_candidate_tactic_sequence(load_config['candidate_tac_path'])
    else:
        trainer.train_candidate_tactic_sequence(args.batch_size)

    if load_config['tuned_tac_path'] != '':
        trainer.load_tuned_candidate_seq(load_config['tuned_tac_path'])
    else:
        trainer.tuner_candidate_seq(args.batch_size)

    if args.without_integration:
        return

    trainer.make_strategy(args.batch_size)


if __name__ == '__main__':
    main()
