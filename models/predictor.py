import torch.nn as nn
from models.residue import PerResidueEncoder
from models.attention import GAEncoder
from models.common import get_pos_CB, construct_3d_basis
from utils.protein import ATOM_N, ATOM_C
from scripts.datasets import *


class ComplexEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.relpos_embedding = nn.Embedding(cfg.max_relpos * 2 + 2, cfg.pair_sequence_feat_dim)
        self.residue_encoder = PerResidueEncoder(cfg.node_feat_dim)
        if config.feature.pass1:
            self.aa_pair1 = nn.Linear(21, cfg.pair_sequence_feat_dim, bias=False)
            self.aa_pair2 = nn.Linear(21, cfg.pair_sequence_feat_dim, bias=False)
        if config.feature.pass2:
            self.layer_norm_pair = nn.LayerNorm(cfg.pair_sequence_feat_dim)
            self.pair_distance_one_hot = nn.Linear(15, cfg.pair_sequence_feat_dim, bias=False)

        if cfg.geomattn is not None:
            self.ga_encoder = GAEncoder(
                node_feat_dim=cfg.node_feat_dim,
                pair_sequence_feat_dim=cfg.pair_sequence_feat_dim,
                num_layers=cfg.geomattn.num_layers,
                spatial_attn_mode=cfg.geomattn.spatial_attn_mode,
            )
        else:
            self.out_mlp = nn.Sequential(
                nn.Linear(cfg.node_feat_dim, cfg.node_feat_dim), nn.ReLU(),
                nn.Linear(cfg.node_feat_dim, cfg.node_feat_dim), nn.ReLU(),
                nn.Linear(cfg.node_feat_dim, cfg.node_feat_dim),
            )

    def forward(self, pos14, aa, seq, chain, mask_atom):
        same_chain = (chain[:, None, :] == chain[:, :, None])
        relpos = (seq[:, None, :] - seq[:, :, None]).clamp(min=-self.cfg.max_relpos,
                                                           max=self.cfg.max_relpos) + self.cfg.max_relpos
        relpos = torch.where(same_chain, relpos, torch.full_like(relpos, fill_value=self.cfg.max_relpos * 2 + 1))

        pair_sequence_feat = self.relpos_embedding(relpos)

        aa_ont_hot = one_hot(aa, bins=list(range(21))).to(pair_sequence_feat)
        pair_sequence_feat = 1 / 16 * (pair_sequence_feat + (
                    self.aa_pair1(aa_ont_hot)[:, None, :, :] + self.aa_pair2(aa_ont_hot)[:, :, None, :]))

        res_feat = self.residue_encoder(aa, pos14, mask_atom)
        CB_pos = get_pos_CB(pos14, mask_atom)
        pair_distance_feat = torch.sqrt(
            ((CB_pos.unsqueeze(2) - CB_pos.unsqueeze(1)) ** 2).sum(-1))

        if config.feature.pass2:
            pair_distance_feat_one_hot = one_hot(pair_distance_feat,
                                                 bins=list(np.linspace(27.0 / 8.0, 171.0 / 8.0, 15)))
            pair_sequence_feat = self.layer_norm_pair(pair_sequence_feat) + self.pair_distance_one_hot(
                pair_distance_feat_one_hot)

        t = pos14[:, :, ATOM_CA]
        R = construct_3d_basis(pos14[:, :, ATOM_CA], pos14[:, :, ATOM_C], pos14[:, :, ATOM_N])
        mask_residue = mask_atom[:, :, ATOM_CA]

        res_feat = self.ga_encoder(R, t, get_pos_CB(pos14, mask_atom),
                                   res_feat,
                                   pair_sequence_feat,
                                   pair_distance_feat,
                                   mask_residue)
        return res_feat


class DDGReadout(nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        contact_dim = 0
        if config.feature.contact_area[0]: contact_dim = 1
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + contact_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.project = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, node_feat_wt, node_feat_mut, contact_area, mask=None):
        feat_wm = torch.cat([node_feat_wt, node_feat_mut], dim=-1)
        feat_mw = torch.cat([node_feat_mut, node_feat_wt], dim=-1)
        feat_diff = self.mlp(feat_wm) - self.mlp(feat_mw)
        per_residue_ddg = self.project(feat_diff).squeeze(-1)
        ddg = per_residue_ddg.sum(dim=1)
        return ddg


class DDGPredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = ComplexEncoder(cfg)
        self.ddG_readout = DDGReadout(cfg.node_feat_dim)

    def forward(self, complex_wt, complex_mut, contact_area=None, ddG_true=None):
        mask_atom_wt = complex_wt['pos14_mask'].all(dim=-1)
        mask_atom_mut = complex_mut['pos14_mask'].all(dim=-1)

        feat_wt = self.encoder(complex_wt['pos14'], complex_wt['aa'], complex_wt['seq'], complex_wt['chain_seq'],
                               mask_atom_wt)
        feat_mut = self.encoder(complex_mut['pos14'], complex_mut['aa'], complex_mut['seq'], complex_mut['chain_seq'],
                                mask_atom_mut)

        mask_res = mask_atom_wt[:, :, ATOM_CA]
        ddG_pred = self.ddG_readout(feat_wt, feat_mut, contact_area, mask_res)
        return ddG_pred