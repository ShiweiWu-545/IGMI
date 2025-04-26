import os
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.SASA import ShrakeRupley

max_asa = {'ALA':  113.0,'ARG':  241.0,'ASN':  158.0,'ASP':  151.0,'CYS':  140.0,
'GLU':  183.0,'GLN':  189.0,'GLY':  85.0,'HIS':  194.0,'ILE':  182.0,'LEU':  180.0,'LYS':  211.0,
'MET':  204.0,
'PHE':	218.0 ,
'PRO':  143.0,
'SER':  122.0,
'THR':  146.0,
'TRP':  259.0,
'TYR':  229.0,
'VAL':  160.0}


def get_protein_divide(pdb_name):
    chains = []
    tem_data = open('/home/xin/UniBind/input/skempi_v2.csv').readlines()
    for line in tem_data:
        if line[0] == '#': continue
        name = line.split(';', 1)[0]
        if pdb_neme_toAB(name) == pdb_name.split('.')[0]:
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

def cal_asa_bio(file, out_dir):

    pdb_name = file.name

    outf =out_dir / (pdb_name[:-4] + "_asa.pdb")
    p = PDBParser(QUIET=1)
    struct = p.get_structure(None, file)
    sr = ShrakeRupley()
    sr.compute(struct, level="R")

    for model in struct:
        for chain in model:
            for residue in chain:

                new_b_factor = round(residue.sasa, 2)
                for atom in residue:
                    atom.set_bfactor(new_b_factor)

    io = PDBIO()

    io.set_structure(struct)

    io.save(str(outf))
    return outf

def determine_region_group(row):
    if row['d_rasa'] == 0:
        if row['rasa_c'] > 0.25:
            return 'SUR'
        else:
            return 'INT'
    elif row['d_rasa'] > 0:
        if row['rasa_m'] < 0.25:
            return 'SUP'
        elif row['rasa_c'] > 0.25:
            return 'RIM'
        else:
            return 'COR'
    else:
        raise('c>m,cal error')

def analyse_asa(asa_c,asa_0 = None,asa_1=None):

    df_c = analyse_pdb(asa_c)
    df_0 = analyse_pdb(asa_0)
    df_0['protein'] = 0
    df_1 = analyse_pdb(asa_1)
    df_1['protein'] = 1
    df_m = pd.concat([df_1,df_0], ignore_index=True)
    df_m.sort_values(by='pos', inplace=True)
    df_m = df_m.reset_index(drop=True)
    df_c.sort_values(by='pos', inplace=True)
    df_c = df_c.reset_index(drop=True)
    assert len(df_c) == len(df_m)
    assert (df_c['res'] == df_m['res']).all()
    df_c['asa_m'] = df_m['asa']
    df_c['protein'] = df_m['protein']
    df_c['max_asa'] = [max_asa[key] for key in df_c['res']]
    df_c['rasa_c'] = df_c['asa'] / df_c['max_asa']
    df_c['rasa_m'] = df_c['asa_m'] / df_c['max_asa']
    df_c['d_rasa'] = df_c['rasa_m'] - df_c['rasa_c']
    df_c['region'] = df_c.apply(determine_region_group,axis=1)
    new_order = ['pos', 'res', 'chain','region','protein','rasa_c','rasa_m','d_rasa','max_asa','asa','asa_m']
    df_c = df_c[new_order]

    tem_parent = os.path.dirname(str(asa_c))
    tem_file = os.path.splitext(os.path.basename(str(asa_c)))[0]
    tem_file = '_'.join(tem_file.split('_')[1:3])
    out_file = os.path.join(tem_parent, tem_file) + '.region'

    df_c.to_csv(out_file,index =False)


def analyse_pdb(asa_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", str(asa_pdb))
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]
    chains = list(model.get_chains())
    b_factors = []
    res_num = []
    res_type = []
    chain_name = []
    for chain in chains:
        for res in chain:
            res_num.append(res.id[1])
            res_type.append(res.resname)
            chain_name.append(chain.id)
            b_factors.append(res['CA'].bfactor)
    df = pd.DataFrame({'pos':res_num,
                       'res':res_type,
                       'chain':chain_name,
                      'asa':b_factors})
    return df


def get_region_2(name,proteinA,proteinB,out_dir):
    pdb_dir_name = name+'_'+(proteinA+proteinB)

    out_dir = out_dir / pdb_dir_name
    pdb = out_dir / (name+'_'+(proteinA+proteinB)+'.pdb')
    pdb_0 = out_dir / (name+'_'+proteinA+'.pdb')
    pdb_1 = out_dir / (name+'_'+proteinB+'.pdb')
    cmd_extract_chains_0 = "/home/xin/anaconda3/bin/pdb_selchain -" + ','.join(proteinA) +' ' + str(pdb) + ' > ' + str(pdb_0)
    cmd_extract_chains_1 = "/home/xin/anaconda3/bin/pdb_selchain -" + ','.join(proteinB) + ' ' + str(pdb) + ' > ' + str(pdb_1)
    os.system(cmd_extract_chains_0)
    os.system(cmd_extract_chains_1)
    asa_0 = cal_asa_bio(pdb_0,out_dir)
    asa_1 = cal_asa_bio(pdb_1,out_dir)
    asa_c = cal_asa_bio(pdb,out_dir)
    analyse_asa(asa_c,asa_0,asa_1)

def get_region_ske(name,proteinA,proteinB):
    name = name + '.pdb'
    out_dir = os.path.dirname(os.getcwd()) + '/data'

    pdb_name = name.split('_')[1]
    out_dir = Path(out_dir)
    pdb = out_dir / name
    pdb_0 = out_dir / (pdb_name+'_'+proteinA+'.pdb')
    pdb_1 = out_dir / (pdb_name+'_'+proteinB+'.pdb')
    cmd_extract_chains_0 = "pdb_selchain -" + ','.join(proteinA) + ' ' + str(pdb) + ' > ' + str(pdb_0)
    cmd_extract_chains_1 = "pdb_selchain -" + ','.join(proteinB) + ' ' + str(pdb) + ' > ' + str(pdb_1)
    os.system(cmd_extract_chains_0)
    os.system(cmd_extract_chains_1)
    asa_0 = cal_asa_bio(pdb_0,out_dir)
    asa_1 = cal_asa_bio(pdb_1,out_dir)
    asa_c = cal_asa_bio(pdb,out_dir)
    analyse_asa(asa_c,asa_0,asa_1)

    os.remove(pdb_0)
    os.remove(pdb_1)
    os.remove(asa_0)
    os.remove(asa_1)
    os.remove(asa_c)
