import torch
from utils.protein import ATOM_CA, ATOM_CB
import torch.nn as nn


def get_pos_CA(pos14, atom_mask):
    pos_CA = pos14[:, :, ATOM_CA]
    return pos_CA


def get_pos_CB(pos14, atom_mask):
    N, L = pos14.shape[:2]
    mask_CB = atom_mask[:, :, ATOM_CB]
    mask_CB = mask_CB[:, :, None].expand(N, L, 3)
    pos_CA = pos14[:, :, ATOM_CA]
    pos_CB = pos14[:, :, ATOM_CB]
    return torch.where(mask_CB, pos_CB, pos_CA)


def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    return (e * v).sum(dim=dim, keepdim=True) * e


class PositionalEncoding(nn.Module):

    def __init__(self, num_funcs=6):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands',
                             2.0 ** torch.linspace(0.0, num_funcs - 1, num_funcs))  # 在模型中增加一个缓冲区,且值为[1,2,4,8,16,32]


def construct_3d_basis(center, p1, p2):
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)
    return mat


def local_to_global(R, t, p):
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)
    q = torch.matmul(R, p) + t.unsqueeze(-1)
    q = q.transpose(-1, -2).reshape(p_size)
    return q


def global_to_local(R, t, q):
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)
    if t is None:
        p = torch.matmul(R.transpose(-1, -2), q)
    else: 
        p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))
    p = p.transpose(-1, -2).reshape(q_size)
    return p