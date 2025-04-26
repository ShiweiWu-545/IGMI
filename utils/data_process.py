import math
import torch
from torch.utils.data._utils.collate import default_collate
from utils.protein import ATOM_CA, parse_pdb
from data.my_config import config
from scripts.datasets import *
import pandas as pd
import copy
from pathlib import Path
import shutil
from utils.cal_rasa import get_region_ske
import os


class PaddingCollate(object):
    def __init__(self, length_ref_key='mutation_mask', pad_values={'aa': 20, 'pos14': float('999'), 'icode': ' ', 'chain_id': '-'}, donot_pad={'foldx'}, eight=False):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.donot_pad = donot_pad
        self.eight = eight

    def _pad_last(self, x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            if value == 0:
                return x
            pad = [value] * (n - len(x))
            return x + pad
        elif isinstance(x, str):
            if value == 0:
                return x
            pad = value * (n - len(x))
            return x + pad
        elif isinstance(x, dict):
            padded = {}
            for k, v in x.items():
                if k in self.donot_pad:
                    padded[k] = v
                else:
                    padded[k] = self._pad_last(v, n, value=self._get_pad_value(k))
            return padded
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        if config.feature.choice_residue == 'Mutant nearby residues':
            max_length = int(config.feature.nearby_residues)
        else:
            max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {}
            for k, v in data.items():
                if k in ('wt', 'mut', 'ddG', 'mutation_mask', 'index', 'mutation', 'pdb_name'):
                    data_padded[k] = self._pad_last(v, max_length, value=self._get_pad_value(k))
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def _mask_list(l, mask):
    return [l[i] for i in range(len(l)) if mask[i]]


def _mask_string(s, mask):
    return ''.join([s[i] for i in range(len(s)) if mask[i]])


def _mask_dict_recursively(d, mask):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
            out[k] = v[mask]
        elif isinstance(v, list) and len(v) == mask.size(0):
            out[k] = _mask_list(v, mask)
        elif isinstance(v, str) and len(v) == mask.size(0):
            out[k] = _mask_string(v, mask)
        elif isinstance(v, dict):
            out[k] = _mask_dict_recursively(v, mask)
        else:
            out[k] = v
    return out


class KnnResidue(object):

    def __init__(self, num_neighbors=config.feature.nearby_residues,
                 max_rnum=config.feature.Residue_selection_based_on_core.max_rnum,
                 mut_lr=config.feature.Residue_selection_based_on_core.mut_lr,
                 core_ratio=config.feature.Residue_selection_based_on_core.core_ratio):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.max_rnum = max_rnum
        self.mut_lr = mut_lr
        self.core_ratio = core_ratio

    def get_mut_around_res(self, pos_CA, pos_CA_mut):
        diff = pos_CA_mut.view(1, -1, 3) - pos_CA.view(-1, 1, 3)
        dist = torch.linalg.norm(diff, dim=-1)
        try:
            index_list = dist.min(dim=1)[0].argsort().tolist()
        except IndexError as e:
            raise e
        return index_list

    def get_center(self, data):
        num_points = data.size(0)
        sum_of_points = torch.sum(data, dim=0)
        center = sum_of_points / num_points
        return center

    def translate_point(self, point1, point2, distance):
        translation_vector = point2 - point1
        vector_length = torch.norm(translation_vector)
        unit_translation_vector = translation_vector / vector_length
        scaled_translation_vector = unit_translation_vector * distance
        translated_point1 = point1 + scaled_translation_vector
        return translated_point1

    def __call__(self, data, core_index,):
        core_mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
        core_mask[core_index] = True

        pos_CA = data['wt']['pos14'][:,
                 ATOM_CA]
        pos_CA_mut = pos_CA[data['mutation_mask']]

        if core_index is None:
            index_list = self.get_mut_around_res(pos_CA, pos_CA_mut)[:self.num_neighbors]
            mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
            mask[index_list] = True
            return _mask_dict_recursively(data,
                                          mask), mask

        index_list_old = self.get_mut_around_res(pos_CA, pos_CA_mut)[:self.num_neighbors]
        index_list = copy.deepcopy(index_list_old)
        core_pos_CA_center = self.get_center(pos_CA[core_mask])
        rnum = 0

        if config.feature.Residue_selection_based_on_core.dynamic_mut:
            debug = (len(set(core_index) - (set(index_list))) / len(set(core_index)))

            new_df = pd.DataFrame()
            new_df['core_ratio'] = [1 - debug]
            new_df['distance(core center-mut)'] = (torch.norm(core_pos_CA_center - pos_CA_mut)).item()

            while (len(set(core_index) - (set(index_list))) / len(set(core_index))) > (1 - self.core_ratio):
                debug = (len(set(core_index) - (set(index_list))) / len(set(core_index)))
                print('不在所选残基中的core占总core的比例：', debug)
                pos_CA_mut = self.translate_point(pos_CA_mut, core_pos_CA_center, self.mut_lr)
                index_list = self.get_mut_around_res(pos_CA, pos_CA_mut)[:self.num_neighbors]
                rnum += 1
                if rnum >= self.max_rnum:
                    break
        else:
            index_list = [x for x in index_list if x not in core_index]
            index_list = list(core_index) + index_list[:(self.num_neighbors - len(list(core_index)))]

        if config.feature.Residue_selection_based_on_core.Residues_outside_the_regulatory_capacity:
            mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
            mask[index_list] = True
        else:
            if rnum >= self.max_rnum:
                mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
                mask[index_list_old] = True
            else:
                mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
                mask[index_list] = True

        return _mask_dict_recursively(data,
                                      mask), mask


def get_mutant_nearby_residues(data_wt, data_mut, core_index):
    transform = KnnResidue()
    mutation_mask = (data_wt['aa'] != data_mut['aa'])
    batch, mask = transform({'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask}, core_index)
    return batch, mask


def choice_residue(data_wt, data_mut, pdb_name):
    core_index = None
    if config.feature.Residue_selection_based_on_core.label:
        _, core_mask = get_core_residues(data_wt, data_mut, pdb_name)
        if core_mask is None:
            core_index = None
        else:
            core_index = torch.nonzero(core_mask).squeeze().tolist()

    batch, mask = get_mutant_nearby_residues(data_wt, data_mut, core_index)
    return batch


def load_wt_mut_pdb_pair(wt_path, mut_path, proteinA, proteinB):
    tem_name = os.path.splitext(os.path.basename(wt_path))[0]
    pdb_name = '_'.join(tem_name.split('_')[1:3])

    get_region_ske(tem_name, proteinA, proteinB)

    try:
        data_wt = parse_pdb(wt_path)
        data_mut = parse_pdb(mut_path)
        batch = choice_residue(data_wt, data_mut, pdb_name)

        return batch
    except IndexError:
        print('No mutation site!')


def get_region(pdb_name):

    if pdb_name.split('_')[0] == 'reverse':
        pdb = '_'.join(pdb_name.split('_')[1: 3])
    elif pdb_name.split('_')[0] == 'HM':
        pdb = '_'.join(pdb_name.split('_')[0: 3])
    else:
        pdb = '_'.join(pdb_name.split('_')[0: 2])

    return_dict = {}
    chain_A_B = get_protein_divide(pdb_name)

    region_dict = {'COR': [],
                   'RIM': [],
                   'INT': [],
                   'SUP': [],
                   'SUR': [], }

    region_dict_proteinA = {'COR': [],
                            'RIM': [],
                            'INT': [],
                            'SUP': [],
                            'SUR': [], }

    region_dict_proteinB = {'COR': [],
                            'RIM': [],
                            'INT': [],
                            'SUP': [],
                            'SUR': [], }

    region_f = pd.read_csv('../data/' + pdb + '.region')

    # 汇总到返回字典上
    return_dict['complex'] = region_dict
    return_dict['protein' + chain_A_B[0]] = region_dict_proteinA
    return_dict['protein' + chain_A_B[1]] = region_dict_proteinB

    for index, line in region_f.iterrows():
        pos = str(line['pos'])
        chain = line['chain']
        reg = line['region']

        region_dict[reg].append(pos)

        if chain in chain_A_B[0]:
            region_dict_proteinA[reg].append(pos)

        elif chain in chain_A_B[1]:
            region_dict_proteinB[reg].append(pos)

    return_dict['complex'] = region_dict
    return_dict['protein' + chain_A_B[0]] = region_dict_proteinA
    return_dict['protein' + chain_A_B[1]] = region_dict_proteinB

    return return_dict


def get_protein_divide(pdb_name):

    chains = []
    tem_data = open('../datasets/skempi_v2.csv').readlines()
    for line in tem_data:
        if line[0] == '#': continue
        name = line.split(';', 1)[0]
        if pdb_neme_toAB(name) == pdb_name.split('_')[0] + '_' + pdb_name.split('_')[1]:
            chains = name.split('_', 1)[1].split('_')
    return chains


def pdb_neme_toAB(i):
    mut = None
    if isinstance(i, list):
        mut = i[2]
        i = i[0]
    tem = i.split('_', 1)
    tem[1] = tem[1].replace('_', '')
    tem = tem[0] + '_' + tem[1]

    if mut != None:
        mut = mut.replace(',', '_')
        pdb = tem + '_' + mut
    else:
        pdb = tem
    return pdb


def get_core_residues(data_wt, data_mut, pdb_name):

    mutation_mask = (data_wt['aa'] != data_mut['aa'])
    data = {'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask}
    if True not in mutation_mask:
        raise IndexError

    region_dict = get_region(pdb_name)
    if region_dict is None:return None, None

    core_resseq = region_dict['complex']['COR']

    mask = torch.zeros([data_wt['resseq'].size(0),], dtype=torch.bool)
    indices = [index for index, value in enumerate(data_wt['resseq'].tolist()) if str(value) in core_resseq]

    mask[indices] = True
    return _mask_dict_recursively(data, mask), mask


def Optimised_compilation(model):
    try:
        model = torch.compile(model)
        torch.set_float32_matmul_precision('high')
    except RuntimeError:
        print('Windows not yet supported for Optimised compilation')
    return model


def make_input(path_pdb, path_mutation, output_dir, return_json=False):
    path_bin_evoef2 = os.path.join(os.path.dirname(os.getcwd()), 'EvoEF2/EvoEF2')

    from utils.tools import ProteinTools
    tools = ProteinTools()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_wt = output_dir / 'wt.pdb'
    path_mut = output_dir / 'mut.pdb'
    path_mut_list = output_dir / 'mutation.txt'

    shutil.copy(path_pdb, path_wt)
    shutil.copy(path_mutation, path_mut_list)
    tools.evoef2(path_wt, path_mut_list, path_mut, path_bin=path_bin_evoef2)

    record = {
        'path_wt': str(path_wt.name),
        'path_mut': str(path_mut.name),
        'mutation': path_mut_list.read_text().strip(';'),
        'dataset': 'test',
    }
    if return_json:
        return record
    else:
        return None