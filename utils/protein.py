import warnings
import torch
from Bio import BiopythonWarning
from Bio.PDB import Selection
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one, three_to_index, is_aa
import re
from data.my_config import config


NON_STANDARD_SUBSTITUTIONS = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

RESIDUE_SIDECHAIN_POSTFIXES = {
    'A': ['B'],
    'R': ['B', 'G', 'D', 'E', 'Z', 'H1', 'H2'],
    'N': ['B', 'G', 'D1', 'D2'],
    'D': ['B', 'G', 'D1', 'D2'],
    'C': ['B', 'G'],
    'E': ['B', 'G', 'D', 'E1', 'E2'],
    'Q': ['B', 'G', 'D', 'E1', 'E2'],
    'G': [],
    'H': ['B', 'G', 'D1', 'D2', 'E1', 'E2'],
    'I': ['B', 'G1', 'G2', 'D1'],
    'L': ['B', 'G', 'D1', 'D2'],
    'K': ['B', 'G', 'D', 'E', 'Z'],
    'M': ['B', 'G', 'D', 'E'],
    'F': ['B', 'G', 'D1', 'D2', 'E1', 'E2', 'Z'],
    'P': ['B', 'G', 'D'],
    'S': ['B', 'G'],
    'T': ['B', 'G1', 'G2'],
    'W': ['B', 'G', 'D1', 'D2', 'E1', 'E2', 'E3', 'Z2', 'Z3', 'H2'],
    'Y': ['B', 'G', 'D1', 'D2', 'E1', 'E2', 'Z', 'H'],
    'V': ['B', 'G1', 'G2'],
}

RESIDUE_ALL_ATOM = {
    'A': ['CB', 'H', 'HA', '1HB', '2HB', '3HB'],
    'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', 'HE', '1HH1', '2HH1', '1HH2', '2HH2'],
    'N': ['CB', 'CG', 'OD1', 'ND2', 'H', 'HA', '1HB', '2HB', '1HD2', '2HD2'],
    'D': ['CB', 'CG', 'OD1', 'OD2', 'H', 'HA', '1HB', '2HB'],
    'C': ['CB', 'SG', 'H', 'HA', '1HB', '2HB', 'HG'],
    'E': ['CB', 'CG', 'CD', 'OE1', 'OE2', 'H', 'HA', '1HB', '2HB', '1HG', '2HG'],
    'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HE2', '2HE2'],
    'G': ['H', '1HA', '2HA'],
    'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', 'H', 'HA', '1HB', '2HB', 'HD2', 'HE1', 'HE2'],
    'I': ['CB', 'CG1', 'CG2', 'CD1', 'H', 'HA', 'HB', '1HG1', '2HG1', '1HG2', '2HG2', '3HG2', '1HD1', '2HD1', '3HD1'],
    'L': ['CB', 'CG', 'CD1', 'CD2', 'H', 'HA', '1HB', '2HB', 'HG', '1HD1', '2HD1', '3HD1', '1HD2', '2HD2', '3HD2'],
    'K': ['CB', 'CG', 'CD', 'CE', 'NZ', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD', '1HE', '2HE', '1HZ', '2HZ', '3HZ'],
    'M': ['CB', 'CG', 'SD', 'CE', 'H', 'HA', '1HB', '2HB', '1HG', '2HG', '1HE', '2HE', '3HE'],
    'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'H', 'HA', '1HB', '2HB', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ'],
    'P': ['CB', 'CG', 'CD', 'HA', '1HB', '2HB', '1HG', '2HG', '1HD', '2HD'],
    'S': ['CB', 'OG', 'H', 'HA', '1HB', '2HB', 'HG'],
    'T': ['CB', 'OG1', 'CG2', 'H', 'HA', 'HB', 'HG1', '1HG2', '2HG2', '3HG2'],
    'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'H', 'HA', '1HB', '2HB', 'HD1', 'HE1', 'HE3', 'HZ2', 'HZ3', 'HH2'],
    'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'H', 'HA', '1HB', '2HB', 'HD1', 'HD2', 'HE1', 'HE2', 'HH'],
    'V': ['CB', 'CG1', 'CG2', 'H', 'HA', 'HB', '1HG1', '2HG1', '3HG1', '1HG2', '2HG2', '3HG2'],
}

RESIDUE_TRUNK_ALL_ATOM = {
    'A': ['H', 'HA'],
    'R': ['H', 'HA'],
    'N': ['H', 'HA'],
    'D': ['H', 'HA'],
    'C': ['H', 'HA'],
    'E': ['H', 'HA'],
    'Q': ['H', 'HA'],
    'G': ['H', '1HA', '2HA'],
    'H': ['H', 'HA'],
    'I': ['H', 'HA'],
    'L': ['H', 'HA'],
    'K': ['H', 'HA'],
    'M': ['H', 'HA'],
    'F': ['H', 'HA'],
    'P': ['HA'],
    'S': ['H', 'HA'],
    'T': ['H', 'HA'],
    'W': ['H', 'HA'],
    'Y': ['H', 'HA'],
    'V': ['H', 'HA'],
}

GLY_INDEX = 5
ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


def augmented_three_to_one(three):
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    return three_to_one(three)


def augmented_three_to_index(three):
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    return three_to_index(three)


def augmented_is_aa(three):
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    return is_aa(three, standard=True)


def is_hetero_residue(res):
    return len(res.id[0].strip()) > 0


def get_atom_name_postfix(atom):
    name = atom.get_name()
    if name in ('N', 'CA', 'C', 'O'):
        return name
    if name[-1].isnumeric():
        return name[-2:]
    else:
        return name[-1:]


def get_residue_pos14_All_non_hydrogen_atoms(res):
    pos14 = torch.full([14, 3], float('inf'))
    suffix_to_atom = {}
    for a in res.get_atoms():
        if re.match('^\d+H.*$', a.fullname.strip()) \
                or re.match('^H.*$', a.fullname.strip()):
            continue
        suffix_to_atom[get_atom_name_postfix(a)] = a

    atom_order = ['N', 'CA', 'C', 'O'] + RESIDUE_SIDECHAIN_POSTFIXES[augmented_three_to_one(res.get_resname())]
    for i, atom_suffix in enumerate(atom_order):
        if atom_suffix not in suffix_to_atom: continue
        pos14[i,0], pos14[i,1], pos14[i,2] = suffix_to_atom[atom_suffix].get_coord().tolist()
    return pos14


def parse_pdb(path, model_id=0):
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    return parse_complex(structure, model_id)


def parse_complex(structure, model_id=None):
    if model_id is not None:
        structure = structure[model_id]
    chains = Selection.unfold_entities(structure, 'C')

    aa, resseq, icode, seq = [], [], [], []
    pos14, pos14_mask = [], []
    chain_id, chain_seq = [], []
    for i, chain in enumerate(chains):
        seq_this = 0
        for res in chain:
            resname = res.get_resname()
            if not augmented_is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue

            chain_id.append(chain.get_id())
            chain_seq.append(i+1)

            restype = augmented_three_to_index(resname)
            aa.append(restype)

            if config.feature.Atom[0] == 'All non-hydrogen atoms':
                pos14_this = get_residue_pos14_All_non_hydrogen_atoms(res)

            pos14_mask_this = pos14_this.isfinite()
            pos14.append(pos14_this.nan_to_num(posinf=99999))
            pos14_mask.append(pos14_mask_this)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_resseq = resseq_this - resseq[-1]
                if d_resseq == 0: seq_this += 1
                else: seq_this += d_resseq
            resseq.append(resseq_this)
            icode.append(icode_this)
            seq.append(seq_this)

    if len(aa) == 0:
        return None

    return {
        'name': structure.get_id(),
        'chain_id': ''.join(chain_id),
        'chain_seq': torch.LongTensor(chain_seq),
        'aa': torch.LongTensor(aa),
        'resseq': torch.LongTensor(resseq),
        'icode': ''.join(icode),
        'seq': torch.LongTensor(seq),
        'pos14': torch.stack(pos14),
        'pos14_mask': torch.stack(pos14_mask),
    }
