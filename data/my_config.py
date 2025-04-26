from easydict import EasyDict as edict


config = edict({
    'debug': {'save_Att_weight':
                  [False, r'D:\Desktop\literature\portein\Deep learning guided optimization of human antibody against SARS_CoV_2 variants with broad neutralization\recurrence\binding-ddg-predictor-main\Data_analysis\data\Model_Result\Weight_analysis'],
              'save_Select_res':
                  [False, r'D:\Desktop\literature\portein\Deep learning guided optimization of human antibody against SARS_CoV_2 variants with broad neutralization\recurrence\binding-ddg-predictor-main\Data_analysis\data\Model_Result\Weight_analysis']},
    'feature': {
        'Atom': ['All non-hydrogen atoms', 14],
        'Atom_list': [['All non-hydrogen atoms', 14],
                      ['All atoms', 24],
                      ['Trunk atoms', 4],
                      ['All trunk atoms', 7],
                      ''],

        'Spatial_distance': 'CB',
        'Spatial_distance_list': ['CA',
                                  'CB'],

        'surface': 48,
        'nearby_residues': 128,

        'choice_residue': 'Mutant nearby residues',
        'choice_residue_list': ['Mutant nearby residues',
                                'Junction surface residues_14',
                                'Junction surface residues_CA',
                                'The intersection of A and B_14',
                                'The intersection of A and B_CA'],

        'contact_area': [False, 'Residue level'],
        'contact_area_list': [[False, 'Residue level'],
                              [True, 'Compound level (MLP)'],
                              [True, 'Compound level (Manual design)']],

        'datasets_extend_reverse': True,
        'Recycling': [False, 3],
        'Dropout': [False, 0.5],
        'relpos_sparse': False,
        'Side_chain_geometry': True,

        'LayerNorm': False,
        'weight': {'SENet': False,
                   'Attention': True},
        'diff_Attention': True,
        'pass0': True,
        'pass1': True,
        'pass2': False,
        'pass3': False,
        'Gate': {'gate0': False,
                 'gate1': False,
                 'gate2': False,
                 'gate3': False,
                 'gate4': False,
                 'gate5': False,
                 },
        'freeze_parameter': False,
        'Residue_selection_based_on_core': {'label': True,
                                            'dynamic_mut': True,
                                            'Residues_outside_the_regulatory_capacity': True,
                                            'max_rnum': 20,
                                            'mut_lr': 2,
                                            'core_ratio': 0.75},

        },


    'model': {
        'pool': True,
        'save_model': True,
        'model': '../data/model.pt',
        'my_model': {
            'path': '../data/mymodel/',

            'model_best': '../data/mymodel/mymodel.pt',
            'train_model': '../data/mymodel/train_model/',
            'train_model_best': '../data/mymodel/'},
        'device': 'cuda',
        'node_feat_dim': 128,
        'pair_sequence_feat_dim': 64,
        'max_relpos': 32,

        'geomattn': {
            'num_layers': 3,
            'spatial_attn_mode': 'CB'}},


    'train': {

            'max_iters': 100000,
            'val_freq': 200,
            'val_num': 50,
            'batch_size': 1,
            'seed': 2021,

            'optimizer': {
                'type': 'adam',
                'lr': 0.000005,
                'weight_decay': 0.0,
                'beta1': 0.9,
                'beta2': 0.999},
    },


    'datasets': {
        'data': '../data/',
        'total': '../datasets/ /total_datasets.pt',
        'train': {
            'dataset_del_S1131': {
                'CSV': '../datasets/skempi_v2_del_S1131.csv',
                'PT': '../datasets/ /dataset-S1131.pt'},
            'dataset_del_M1707': {
                'CSV': '../datasets/skempi_v2_del_M1707.csv',
                'PT': '../datasets/ /dataset-M1707.pt'},
            'no_AbAg': {'CSV': '',
                        'PT': '../datasets/ /no_AbAg.pt'}},

        'val': {
            'dataset_path': ''},
        'test': {
            'S1131': '../datasets/ /S1131.pt',
            'M1707': '../datasets/ /M1707.pt',
            'S645': '../datasets/ /S645.pt',
            'S645_no27': '../datasets/ /S645_no27.pt',
            'predict': '../datasets/ /predict_data.pt',
            'C3684': '../datasets/ /C3684.pt',
            'HIV_78': '../datasets/ /HIV_78_100-Binding affinity relative to WT.pt',
            'AB_228': '../datasets/ /AB_228.pt'},
        'contact_area': '../datasets/contact_area.surface'},


    'observer': {
        'parameter': {
            'parameter_path': '../procedure_parameter/parameter'},
        'train_losses_epoch': True,
        'train_Rp': True,
        'validation_Rp': False,
        'validation_losses_epoch': False,
        'test_Rp': True,
        'test_losses_epoch': True}})
