import torch.nn as nn
import torch.nn.functional as F
from scripts.datasets import *
from .common import mask_zero, global_to_local, normalize_vector


def _alpha_from_logits(logits, mask, inf=1e5):

    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)
    
    logits = torch.where(mask_pair, logits, logits-inf)
    alpha = torch.softmax(logits, dim=2)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    s = list(x.size())[:-1] + [n_heads, n_ch]  # [1,128,12,16]
    return x.view(*s)


class GeometricAttention(nn.Module):

    def __init__(self, node_feat_dim, pair_sequence_feat_dim, spatial_attn_mode='CB', value_dim=16, query_key_dim=16, num_query_points=8, num_value_points=8, num_heads=12):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_sequence_feat_dim = pair_sequence_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads
        self.spatial_attn_mode = spatial_attn_mode

        self.proj_query = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=False)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=False)
        self.proj_value = nn.Linear(node_feat_dim, value_dim*num_heads, bias=False)

        self.proj_pair_bias = nn.Linear(pair_sequence_feat_dim, num_heads, bias=False)  # 输入64维，输出12维，不带偏执项

        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)), requires_grad=True)  # 这个函数就是定义可训练参数，并进行初始化

        if config.feature.Gate.gate3:
            self.pair_distance_gateInput = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=99.0, requires_grad=True))  # 这个函数就是定义可训练参数，并进行初始化

        if config.feature.pass3 and config.feature.Side_chain_geometry:
            self.out_transform = nn.Linear(
                in_features=(num_heads*pair_sequence_feat_dim) + (num_heads*value_dim) + (num_heads*(3+3+1)) + (num_heads*pair_sequence_feat_dim),  # 输入(12*64)+(12*16)+(12*7)
                out_features=node_feat_dim,
            )
        elif (config.feature.pass3 == False) and config.feature.Side_chain_geometry:
            self.out_transform = nn.Linear(
                in_features=(num_heads * pair_sequence_feat_dim) + (num_heads * value_dim) + (num_heads * (3 + 3 + 1)),
                out_features=node_feat_dim,
            )
        elif config.feature.pass3 and (config.feature.Side_chain_geometry == False):
            self.out_transform = nn.Linear(
                in_features=(num_heads*pair_sequence_feat_dim) + (num_heads*value_dim) + (num_heads*pair_sequence_feat_dim),
                out_features=node_feat_dim,
            )
        elif (config.feature.pass3 == False) and (config.feature.Side_chain_geometry == False):  # 00
            self.out_transform = nn.Linear(
                in_features=(num_heads*pair_sequence_feat_dim) + (num_heads*value_dim),
                out_features=node_feat_dim,
            )

        self.layer_norm_node = nn.LayerNorm(node_feat_dim)
        if config.feature.LayerNorm:
            self.layer_norm_pair = nn.LayerNorm(pair_sequence_feat_dim)
            self.layer_norm_distance0 = nn.LayerNorm(pair_sequence_feat_dim)

        self.weight_MLP = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim * 2),
            nn.ReLU(),
            nn.Linear(node_feat_dim * 2, node_feat_dim),
        )
        self.weight_abc = nn.Sequential(
            nn.Linear(self.num_heads * self.query_key_dim, node_feat_dim),
            nn.ReLU(),
            nn.Linear(node_feat_dim, 1),
            nn.Softplus()
        )
        self.weight_q = nn.Linear(node_feat_dim, self.num_heads * self.query_key_dim, bias=False)
        self.weight_k = nn.Linear(node_feat_dim, self.num_heads * self.query_key_dim, bias=False)
        self.weight_v = nn.Linear(node_feat_dim, self.num_heads * self.query_key_dim, bias=False)

        self.weight_LayerNorm = nn.LayerNorm(self.num_heads * self.query_key_dim)  # 对x进行标准化

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)
        query_l = query_l.permute(0, 2, 1, 3)
        key_l = key_l.permute(0, 2, 3, 1)
        logits = 1 / np.sqrt(query_l.size()[-1]) * torch.matmul(query_l, key_l)

        logits = logits.permute(0, 2, 3, 1)

        return logits

    def _pair_logits(self, z):
        logits_pair_sequence = self.proj_pair_bias(z)
        return logits_pair_sequence

    def _distance_logits(self, d):
        gamma = F.softplus(self.spatial_coef)
        d = (d ** 2)[:, :, :, None].expand(-1, -1, -1, self.num_heads)
        logtis_distance = d * ((-1 * gamma * np.sqrt(2 / 9)) / 2)

        return logtis_distance

    def _beta_logits(self, R, t, p_CB):
        N, L, _ = t.size()
        qk = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3)
        sum_sq_dist = ((qk.unsqueeze(2) - qk.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logtis_beta = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / 9)) / 2)
        return logtis_beta

    def _pair_aggregation(self, alpha, z):

        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)
        feat_p2n = feat_p2n.sum(dim=2)

        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)
        feat_node = feat_node.sum(dim=2)
        return feat_node.reshape(N, L, -1)

    def _beta_aggregation(self, alpha, R, t, p_CB, x):
        N, L, _ = t.size()
        v = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3)
        aggr = alpha.reshape(N, L, L, self.num_heads, 1) * v.unsqueeze(1)
        aggr = aggr.sum(dim=2)
        feat_points = global_to_local(R, t, aggr)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward_beta(self, R, t, p_CB, x, z, d, mask):
        # Attention logits
        logits_node = self._node_logits(x)
        logits_pair_sequence = self._pair_logits(z)
        logits_pair_distance = self._distance_logits(d)
        # weight系数计算
        a, b, c = self.weight_Attention(logits_node, logits_pair_sequence, logits_pair_distance)

        logits_sum = np.sqrt(1 / 3) * (a * logits_node + b * logits_pair_sequence + c * logits_pair_distance)

        alpha = _alpha_from_logits(logits_sum, mask)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        if config.feature.Side_chain_geometry:
            feat_spatial = self._beta_aggregation(alpha, R, t, p_CB, x)

        # Finally
        if (config.feature.pass3 == False) and config.feature.Side_chain_geometry:  # 01
            feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))  # (N, L, F)
        elif (config.feature.pass3 == False) and (config.feature.Side_chain_geometry == False):  # 00
            feat_all = self.out_transform(torch.cat([feat_p2n, feat_node], dim=-1))  # (N, L, F)

        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_node(x + feat_all)
        return x_updated

    def weight_Attention(self, node, pair, distance):
        x = self.weight_process(node, pair, distance)
        q = _heads(self.weight_q(x), self.num_heads, self.value_dim)
        k = _heads(self.weight_k(x), self.num_heads, self.value_dim)
        v = _heads(self.weight_v(x), self.num_heads, self.value_dim)

        weight_x = 1 / np.sqrt(self.value_dim) * torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        weight_x = torch.softmax(weight_x, dim=2)
        y = torch.matmul(weight_x, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(x.size()[0], x.size()[1], -1)
        y = self.weight_abc(self.weight_LayerNorm(y)).squeeze(dim=2)
        return torch.mean(y[:, 0]), torch.mean(y[:, 1]), torch.sigmoid(torch.mean(y[:, 2]))

    def weight_process(self, node, pair, distance):
        node = node.sum(dim=3).sum(dim=2)[:, None, :]
        pair = pair.sum(dim=3).sum(dim=2)[:, None, :]
        distance = distance.sum(dim=3).sum(dim=2)[:, None, :]
        return torch.cat([node, pair, distance], dim=1)

    def forward(self, R, t, p_CB, x, z, d, mask):
        return self.forward_beta(R, t, p_CB, x, z, d, mask)


class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_sequence_feat_dim, num_layers, spatial_attn_mode='CB'):
        super().__init__()
        self.blocks = nn.ModuleList([
            GeometricAttention(node_feat_dim, pair_sequence_feat_dim, spatial_attn_mode=spatial_attn_mode)
            for _ in range(num_layers)
        ])

    def GeometricAttention_blocks(self, R, t, p_CB, x, z, d, mask, output):

        x = x + output
        for block in self.blocks:
            x = block(R, t, p_CB, x, z, d, mask)
        return x

    def forward(self, R, t, p_CB, x, z, d, mask):
        output = 0
        x = self.GeometricAttention_blocks(R, t, p_CB, x, z, d, mask, output)
        return x