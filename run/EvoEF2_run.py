from utils.data_process import *
from pathlib import Path


wt_path = Path('../data/1PPF_EI.pdb')
path_mutation = Path('../data/individual_list.txt')
out_path = Path('../data')

make_input(wt_path, path_mutation, out_path)