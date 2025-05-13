import sys
sys.path.append('../..')

from sklearn.cluster import SpectralClustering
from sirismt.tools import objects
from sirismt.combiner.prob_dd import BenchmarkSeqReward
import numpy as np
import copy


def get_ind(M):
    ar = np.array(M).T
    ind = [ar[i].sum() for i in range(ar.shape[1])]
    return ind


def create_collaboration_matrix(tac_seqs: list, all_tactics):
    K = len(all_tactics)
    M = [[1 for _ in range(K)] for _ in range(K)]
    if len(tac_seqs) == 0:
        return M

    for seq in tac_seqs:
        if seq is None or len(seq) < 2:
            continue

        for i in range(len(seq) - 1):
            index_0 = all_tactics.index(seq[i].s if not isinstance(seq[i], str) else seq[i])
            index_1 = all_tactics.index(seq[i + 1].s if not isinstance(seq[i], str) else seq[i + 1])
            M[index_0][index_1] += 10

    R = [[1 for _ in range(K)] for _ in range(K)]
    for i in range(K):
        for j in range(i, K):
            if M[i][j] > M[j][i]:
                R[i][j] += 10

    return M


def make_all_tactics(tac_seqs):
    res = set()
    for seq in tac_seqs:
        if seq is None or len(seq) < 1:
            continue
        for tac in seq:
            res.add(tac.s if not isinstance(tac, str) else tac)
    return list(res)


def algorithm1(tac_seqs, c=8):
    all_tactics = make_all_tactics(tac_seqs)
    K = len(all_tactics)
    # Initialize all elements of the collaboration matrix M with 0
    # And build Matrix M
    M = create_collaboration_matrix(tac_seqs, all_tactics)

    # Construct a dependency graph G according to the collaboration matrix M
    G = copy.deepcopy(M)
    for i in range(K):
        for j in range(i, K):
            G[i][j] = G[j][i] = G[i][j] + G[j][i]

    # Segment the dependency graph G into c subgraphs using a graph cut algorithm
    clustering = SpectralClustering(
        n_clusters=c,
        affinity='precomputed',
        assign_labels='discretize',
        random_state=0,
    ).fit(np.array(G))
    labels = clustering.labels_

    # Stretch each subgraph as a pass subsequence with a depth-first traversal algorithm
    ind = get_ind(M)
    subseq_set = []
    visit = [False for _ in range(K)]
    for i in range(c):
        subgraph = [idx for idx in range(K) if labels[idx] == i]
        root = -1
        for candidate in subgraph:
            if root == -1 or ind[candidate] < ind[root]:
                root = candidate

        if root != -1:
            # clear visit array
            for vertex in subgraph:
                visit[vertex] = False
            subseq = []

            def dfs(u):
                subseq.append(u)
                visit[u] = True
                next_v = -1
                for v in range(K):
                    if labels[u] == labels[v] and not visit[v]:
                        if next_v == -1 or M[u][v] > M[u][next_v]:
                            next_v = v
                if next_v != -1:
                    dfs(next_v)

            dfs(root)
            subseq = [objects.Tactic(all_tactics[it]) for it in subseq]
            subseq_set.append(subseq)

    return subseq_set


class FISTree:
    def __init__(self, tac_seqs=None):
        if tac_seqs is None:
            tac_seqs = []
        self.tree = {}
        for tac_seq in tac_seqs:
            self.build(tac_seq)

    def build(self, tac_seq):
        if len(tac_seq) == 0 or tac_seq is None:
            return
        if tac_seq[0].s not in self.tree:
            self.tree[tac_seq[0].s] = FISTree()
        self.tree[tac_seq[0].s].build(tac_seq[1:])

    def count_fit(self, tac_seq, seq_pos):
        if seq_pos >= len(tac_seq):
            return 0
        if tac_seq[seq_pos] in self.tree:
            return 1 + self.tree[tac_seq[seq_pos]].count_fit(tac_seq, seq_pos+1)
        return 0

    def dfs_all(self):
        res = []
        for tac in self.tree:
            res.extend([[tac] + seq for seq in self.tree[tac].dfs_all()])
        if not res:
            res = [[]]
        return res

    def __mutate(self, tac_seq, seq_pos):
        if seq_pos >= len(tac_seq):
            return self.dfs_all()
        if tac_seq[seq_pos] in self.tree:
            return [[tac_seq[seq_pos]] + seq for seq in self.tree[tac_seq[seq_pos]].__mutate(tac_seq, seq_pos+1)]
        return [seq + tac_seq[seq_pos:] for seq in self.dfs_all()]

    def mutate(self, tac_seq, instances, boost_num=4):
        baseline = BenchmarkSeqReward(tac_seq, instances).reward
        tac_seq = [tac.s if not isinstance(tac, str) else tac for tac in tac_seq]
        mutate_pos = [(self.count_fit(tac_seq, tac_i), tac_i) for tac_i, tac in enumerate(tac_seq) if tac in self.tree]
        mutate_pos.sort()
        mutate_pos = mutate_pos[max(0, (len(mutate_pos)-boost_num)):]

        result = []
        for _, seq_pos in mutate_pos:
            result.extend(self.__mutate(tac_seq, seq_pos))
        # result = [[objects.Tactic(tac) for tac in tac_seq] for tac_seq in result]

        def get_reward(ts, insts):
            bsr = BenchmarkSeqReward(ts, insts)
            return bsr.reward

        result = [ts for ts in result if get_reward(ts, instances) < baseline]
        return result


def collect_subseqs(tac_seqs, least=2, most=5):
    tac_seqs = [eval(tac_seq) for tac_seq in tac_seqs]
    all_tactics = make_all_tactics(tac_seqs)
    K = len(all_tactics)

    if K == 0:
        return None
    res = []

    for c in range(int(K/most), int(K/least)):
        if c == 0:
            continue
        res.extend(algorithm1(tac_seqs, c))

    res.sort(key=lambda x: str([str(t) for t in x]))
    n_lst = []
    for lst_i in range(len(res)):
        if len(res[lst_i]) > most or len(res[lst_i]) < least:
            continue
        if (res[lst_i] is not None and lst_i == 0 or
                str([str(ls) for ls in res[lst_i]]) != str([str(ls) for ls in res[lst_i - 1]])):
            n_lst.append(res[lst_i])
    return FISTree(res)
