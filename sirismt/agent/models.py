import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as f
from torch_geometric.nn import GATv2Conv, JumpingKnowledge, global_max_pool, global_mean_pool, \
    global_add_pool, global_sort_pool, AttentionalAggregation


class ReluDNN(nn.Module):
    def __init__(self, num_probes: int, num_tactic: int):
        super(ReluDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_probes, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, num_tactic),
            nn.ReLU()
        )
        self.loss = nn.SmoothL1Loss()

    def forward(self, input_data: torch.Tensor):
        return self.net(input_data)

    def do_train(self, input_data: torch.Tensor, act: torch.Tensor, target: torch.Tensor):
        self.train(True)
        output = self.forward(input_data)

        a_q_values = torch.gather(input=output, index=act, dim=1)

        res_loss = self.loss(a_q_values, target)
        res_loss.backward()

    def predict(self, input_data: torch.Tensor):
        with torch.no_grad():
            self.eval()
            output = self.forward(input_data)
        return output


class GAT(torch.nn.Module):
    def __init__(self, passes, inputLayerSize, outputLayerSize, numAttentionLayers, mode, pool, k, dropout,
                 shouldJump=True):
        super(GAT, self).__init__()
        self.passes = passes  # time steps
        self.mode = mode
        self.k = 1
        self.shouldJump = shouldJump

        self.gats = nn.ModuleList(
            [GATv2Conv(inputLayerSize, inputLayerSize, heads=numAttentionLayers, concat=False, dropout=0, edge_dim=1)
             for _ in range(passes)])
        if self.passes and self.shouldJump:
            self.jump = JumpingKnowledge(self.mode, channels=inputLayerSize, num_layers=self.passes)
        if self.mode == 'cat' and self.shouldJump:
            fcInputLayerSize = ((self.passes + 1) * inputLayerSize * self.k) + 1
        else:
            fcInputLayerSize = (inputLayerSize * self.k) + 1

        if pool == "add":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "sort":
            self.pool = global_sort_pool
            self.k = k
        elif pool == "attention":
            self.pool = AttentionalAggregation(
                gate_nn=nn.Sequential(torch.nn.Linear(fcInputLayerSize - 1, 1), nn.LeakyReLU()))
        else:
            raise ValueError("Not a valid pool")

        self.fc1 = nn.Linear(fcInputLayerSize, fcInputLayerSize // 2)
        self.fc2 = nn.Linear(fcInputLayerSize // 2, fcInputLayerSize // 2)
        self.fcLast = nn.Linear(fcInputLayerSize // 2, outputLayerSize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, problemType, batch):
        if self.passes:
            xs = []
            if self.shouldJump:
                xs = [x]

            for gat in self.gats:
                out = gat(x, edge_index, edge_attr=edge_attr)
                x = f.leaky_relu(out)
                if self.shouldJump:
                    xs += [x]

            if self.shouldJump:
                x = self.jump(xs)

        x = self.pool(x, batch)

        x = torch.cat((x.reshape(problemType.size(0), -1), problemType.unsqueeze(1)), dim=1)
        x = self.fc1(self.dropout(x))
        x = f.leaky_relu(x)
        x = self.fc2(self.dropout(x))
        x = f.leaky_relu(x)
        x = self.fcLast(self.dropout(x))
        return x


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, gnn_passes=2, gnn_numAttentionLayers=5, gnn_inputLayerSize=67,
                 gnn_outputLayerSize=64, gnn_mode='cat', gnn_pool='attention', gnn_k=20, gnn_dropout=0,
                 gnn_shouldJump=True):
        super(DQN, self).__init__()
        self.gnn = GAT(passes=gnn_passes, numAttentionLayers=gnn_numAttentionLayers, inputLayerSize=gnn_inputLayerSize,
                       outputLayerSize=gnn_outputLayerSize, mode=gnn_mode, pool=gnn_pool, k=gnn_k, dropout=gnn_dropout,
                       shouldJump=gnn_shouldJump)
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, graphs):
        gnn_x, gnn_edges, gnn_edge_attr, gnn_problemType, gnn_batch = graphs.x, graphs.edge_index, graphs.edge_attr, graphs.problemType, graphs.batch
        x = torch.cat((F.relu(self.layer1(x)), self.gnn(gnn_x, gnn_edges, gnn_edge_attr, gnn_problemType, gnn_batch)),
                      dim=1)
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNWithoutGNN(nn.Module):

    def __init__(self, n_observations, n_bow_size, n_actions):
        super(DQNWithoutGNN, self).__init__()

        self.layer1_ob = nn.Linear(n_observations, 64)
        self.layer1_bow = nn.Linear(n_bow_size, 64)

        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x, bows):
        x = torch.cat((F.relu(self.layer1_ob(x)), F.relu(self.layer1_bow(bows))), dim=1)
        x = F.relu(self.layer2(x))
        return self.layer3(x)