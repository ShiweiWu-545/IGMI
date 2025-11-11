# IGMI

Enhancing Mutation Impact Prediction in Protein-Protein Interactions through Interpretable Graph-Based Multi-Level Feature Interactions

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
  - [Predicting DDG Using Private Protein Complexes](#predicting-ddg-using-private-protein-complexes)
  - [Examples](#examples)
  - [More examples](#more-examples)
- [Datasets](#datasets)
- [Data Preprocessing](#data-preprocessing)
  - [1. Generate Mutant Structures](#1-generate-mutant-structures)
  - [2. Generate protein sequence embedding vectors](#2-generate-protein-sequence-embedding-vectors)
  - [3. Generate Region Files](#3-generate-region-files)
  - [4. Generate Dataset Files](#4-generate-dataset-files)
- [Model Training](#model-training)
  - [Experimental Setup](#experimental-setup)
  - [Training Example (S1131_SSCV)](#training-example-s1131_sscv)
  - [Model Evaluation and Visualization](#model-evaluation-and-visualization)
- [Training using Large-scale Private Datasets](#training-using-large-scale-private-datasets)
  - [Data Preprocessing](#data-preprocessing-1)
    - [1. Generate Mutant Structures](#1-generate-mutant-structures-1)
    - [2. Generate Protein Embeddings](#2-generate-protein-embeddings-1)
    - [3. Generate Region Files](#3-generate-region-files-1)
    - [4. Generate Dataset Files](#4-generate-dataset-files-1)
  - [Training](#training)
- [Configuration](#configuration)
- [FAQ](#faq)
- [Contact](#contact)
- [Affiliations](#affiliations)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Keywords](#keywords)

## Overview

This repository contains the official implementation of IGMI (Interpretable Graph-based Model for Interactions), a novel interpretable graph-based multi-level feature interaction model for predicting changes in protein-protein binding free energy (ΔΔG) upon mutation. Unlike existing approaches that treat sequence and structure as independent inputs, IGMI explicitly models their interdependencies by integrating multi-dimensional features (1D sequence distance maps, 2D structural distance maps, and 3D coordinates) and multi-scale features (residue- and atomic-based representations).

IGMI consists of two core components:
- **ProteoMAE (Protein Multidimensional Residue Feature Aggregation and Excitation Attention)**: Captures global residue dependencies across spatial dimensions through global messaging and adaptive recalibration mechanisms
- **BackSideAttention (Backbone-Sidechain Attention Mechanism)**: Enhances residue-to-atom level modeling by explicitly incorporating side-chain conformational information

![IGMI Framework Overview](./data/Fig.%201%20Overview%20of%20the%20IGMI%20framework%20for%20predicting%20protein%E2%80%93protein%20binding%20affinity%20changes.png)

## Key Features

- **Multi-level Feature Integration**: Explicitly models interdependencies between multi-dimensional features (1D sequence, 2D structural distance maps, 3D coordinates) and multi-scale features (residue- and atomic-level)
- **ProteoMAE Module**: Global residue dependency modeling through adaptive recalibration mechanisms across spatial dimensions
- **BackSideAttention Module**: Fine-grained residue-atom interaction encoding with side-chain conformational information
- **State-of-the-art Performance**: Achieves superior Pearson correlation and lower RMSE on multiple benchmark datasets
- **Interpretability**: Provides both macro-level (residue interaction networks) and micro-level (atomic interactions) interpretability
- **Robust Generalization**: Effective performance on blind tests with independent protein complexes
- **Pre-trained Protein Embeddings**: Integration with ProtT5 for rich sequence representations

## Installation

### Prerequisites

**Hardware Requirements**:
- Processor: 13th Gen Intel® Core™ i9-13900K × 32
- Graphics: NVIDIA RTX 4090
- Memory: 128GB RAM
- Storage: 7.1TB Available Space
- Operating System: Ubuntu 22.04.2 LTS

**Software Requirements**:
- Python 3.11
- torch==2.9.0
- transformers==4.57.1
- biopython==1.8.1
- MDAnalysis==2.10.0
- numpy==2.3.4
- pandas==2.3.3
- scipy==1.16.3
- matplotlib==3.10.7
- seaborn==0.13.2
- tensorboard==2.20.0

See `requirements.txt` for complete dependency list. Additionally, to complete the environment setup, please download the supplementary file package ([prottrans.zip](https://zenodo.org/records/17563574/files/prottrans.zip?download=1)) from Zenodo, which contains the pre-trained ProtTrans model used for sequence encoding. This file was not uploaded to GitHub due to its large size. After downloading, unzip it and place the extracted folder into the `/IGMI` directory before running any training or evaluation scripts.


### Environment Setup

```bash
# Clone the repository
git clone https://github.com/ShiweiWu-545/IGMI.git
cd IGMI

# Create conda environment
conda create -n IGMI python=3.11 -y
conda activate IGMI

# Install dependencies
pip install -r requirements.txt
```

The environment setup process takes approximately 10 minutes and has been tested by users with limited computational experience.

## Quick Start

### Predicting DDG Using Private Protein Complexes

Model weights `/data/model_weight_predict.pt`, trained using the entire `SKEMPI2 dataset`. Can be directly applied to predict the effect of mutations on affinity for private protein complexes.

```bash
python run_predict.py \
  --wt_path [wt.pdb]
  --mut_path [mut.pdb] \  
  --proteinA [partnerA] \  
  --proteinB [partnerB]
```

[wt.pdb] represents the wild-type complex structure, [mut.pdb] represents the corresponding mutant complex structure, [partnerA] describes the chain of constituent protein A, and [partnerB] describes the chain of constituent protein B.

### Examples

```bash
python run_predict.py \
  --wt_path ./data/Example/ABCD_AHL_TH28R/WT_ABCD_AHL_TH28R.pdb \
  --mut_path ./data/Example/ABCD_AHL_TH28R/MUT_ABCD_AHL_TH28R.pdb \
  --proteinA HL \
  --proteinB A
```

### More examples

Additional examples are provided in the `/data/Example` directory:
```
1A4Y_AB_DA435A    A B
1PPF_EI_EI19H     E I
1YQV_HLY_RY68K_TY40S_IY55V_SY91T    HL Y
1YY9_CDA_SC26D    CD A
2AK4_ABCDE_YE31A  ABC DE
2B2X_HLA_EH64K    HL A
ABCD_AHL_TH28R    HL A
```

## Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| [S1131](./datasets/S1131.csv) | 1,131 | SKEMPI 2.0 subset. Includes 1,131 non-redundant interface single-point mutations |
| [M1707](./datasets/M1707.csv)  | 1,707 | SKEMPI 2.0 subset |
| [S4169](./datasets/S4169.csv)  | 4,169 | SKEMPI 2.0 subset |
| [S8338](./datasets/S8338.csv)  | 8,338 | S4169 Add corresponding reverse mutation |
| [skempi_v2](./datasets/skempi_v2.csv)  | 7085 | Currently the largest manually curated dataset of affinity changes in protein-protein interactions caused by mutations |
| [skempi_v2_del_S1131](./datasets/skempi_v2_del_S1131.csv)  | 5064 | Independent of the S1131 dataset |
| [skempi_v2_del_M1707](./datasets/skempi_v2_del_M1707.csv) | 4856 | Independent of the M1707 dataset |

Download the [Skempi2 structural database](https://zenodo.org/records/17563574/files/Skempi2_ddg_useful.zip?download=1) from `Zenodo`, which contains both wild-type and mutant structures. After unzipping, rename the file to `Skempi2_ddg_useful` and place it in the `./IGMI` directory.

## Data Preprocessing

### 1. Generate Mutant Structures

Use the `Rosetta` suite to generate mutant structures. For detailed instructions, see the [Rosetta documentation](https://docs.rosettacommons.org/docs/latest/cartesian-ddG). The complete mutation structure data used in this paper is available for direct download on `Zenodo`. For details, see Datasets. Extract the downloaded structure database, then rename it to `Skempi2_ddg_useful` and place it in the `./IGMI` directory.

### 2. Generate protein sequence embedding vectors

```bash
cd ./prottrans/
python prottrans.py --input_dir ../Skempi2_ddg_useful
```

### 3. Generate Region Files

```bash
cd ../scripts/
python get_region.py --input_dir ../Skempi2_ddg_useful
```

### 4. Generate Dataset Files

Here, we use the `S1131` dataset as an example. For more personalized usage methods, see `Training using large-scale private datasets`.

```bash
python get_datapt.py \
  --x_data_available ../Skempi2_ddg_useful \
  --x_data ../datasets/S1131.csv \
  --y_data ../datasets/total_ddg_S1_S2_realT.csv \
  --dataset_name S1131
```

Final preprocessing output: `/datasets/Mutant nearby residues/S1131.pt`

## Model Training

### Experimental Setup

We evaluate IGMI under three experimental paradigms:

- **Split-by-Structure Cross-Validation (SSCV)**: S1131_SSCV, S4169_SSCV, S8338_SSCV, M1707_SSCV
- **Cross-Validation (CV)**: S1131_CV, S4169_CV, S8338_CV, M1707_CV  
- **External Validation (EV)**: S1131_EV, M1707_EV

### Training Example (S1131_SSCV)
```bash
# Train 10-fold SSCV models
./run_train_batch.sh S1131_SSCV
```

### Model Evaluation and Visualization

```bash
cd ../procedure_parameter/
python observer.py --label S1131_SSCV
```

The scatter plot of S1131_SSCV results is stored in `./result/S1131_structure/yddg_yhat_big.pdf`, and the CSV file is stored in `./result/S1131_structure/yddg_yhat.csv`. The results are as follows:

![S1131_SSCV Results](./data/S1131_SSCV.png)

```bash
# S1131_SSCV Prediction Results
# yddg_yhat.csv
MSE:2.7201412189125396
RMSE:1.6492850629629008
MAE:1.1927783764112065
R2:0.6030006758863713
Rp:0.7888304693748307
Rp_confidence_interval:[0.7656639320594762, 0.8099532660617836]
P_value:3.1671447904208927e-239
name;yddg;yhat
1IAR_AB_QA8R;0.04216527193784714;-0.3418720066547394
1IAR_AB_FA82D;-0.6289988160133362;-0.12398726493120193
1IAR_AB_TA6D;1.5088087320327759;0.0076317316852509975
...
```

## Training using large-scale private datasets

### Data Preprocessing

#### 1. Generate Mutant Structures

Use the Rosetta suite to generate mutant structures. For detailed instructions, see the [Rosetta documentation](https://docs.rosettacommons.org/docs/latest/cartesian-ddG). A database of pre-generated mutation structures is located in `./Preprocessing/Example/` for simulating private datasets.

#### 2. Generate protein sequence embedding vectors

```bash
cd ./prottrans/
python prottrans.py --input_dir ../Preprocessing/Example
```

#### 3. Generate Region Files

```bash
cd ../scripts/
python get_region.py --input_dir ../Preprocessing/Example
```

#### 4. Generate Dataset Files

```bash
python get_datapt.py \
  --x_data_available ../Preprocessing/Example \
  --x_data ../Preprocessing/private_datasets.csv \
  --y_data ../Preprocessing/private_datasets_yddg.csv \
  --dataset_name privateDatasets
```

The `../Preprocessing/Example` directory contains the structural database for the private dataset, `../Preprocessing/private_datasets.csv` holds the data entries to be indexed, and `private_datasets_yddg.csv` contains the corresponding label values for the indexed entries. Final preprocessing output: `/datasets/Mutant nearby residues/private_datasets.pt`.

### Training

For the training section, refer to the Model Training subsection.

## Configuration

Key model parameters can be configured in `data/my_config.py`:

```python
config = {
    'model': {
        'node_feat_dim': 128,
        'pair_sequence_feat_dim': 64,
        'geomattn.num_layers': 3
    },
    'feature': {
        'surface': 64,
        'nearby_residues': 128,
        'Spatial_distance': 'CB'
    },
    'train': {
        'batch_size': 32,
        'max_iters': 30000,
        'lr': 0.00005
    }
}
```

## FAQ

**Q: Can I run the model without a GPU?**
A: Yes, but prediction will be significantly slower (10-100x depending on the system).

**Q: What PDB formats are supported?**
A: Standard PDB format with complete atom information for both wild-type and mutant structures.

## Contact

For questions, issues, or collaborations, please contact the authors:

- **Shiwei Wu**: wushiwei@hrbeu.edu.cn
- **Weixing Feng**: fengweixing@hrbeu.edu.cn
- **Lei Yu**: yulei@nbic.ecnu.edu.cn  
- **Chengkui Zhao**: zhaochengkui@hrbeu.edu.cn

## Affiliations

- **College of Intelligent Systems Science and Engineering, Harbin Engineering University, Harbin, China**
- **Institute of Biomedical Engineering and Technology, Shanghai Engineering Research Center of Molecular Therapeutics and New Drug Development, School of Chemistry and Molecular Engineering, East China Normal University, Shanghai, China**
- **Shanghai Unicar-Therapy Bio-medicine Technology Co., Ltd, Shanghai, China**

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{Wu2025IGMI,
  title={Enhancing Mutation Impact Prediction in Protein-Protein Interactions through Interpretable Graph-Based Multi-Level Feature Interactions},
  author={Wu, Shiwei and Xu, Nan and Xin, Xiaohui and Zhang, Min and Liu, Haoliang and Zhu, Hongjia and Wei, Zhenyu and Zhao, Chengkui and Yu, Lei and Feng, Weixing},
  note={Manuscript under review},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](./data/LICENSE.txt) file for details.

## Acknowledgments

We thank the SKEMPI consortium for providing the benchmark dataset and the developers of ProtT5 for the pre-trained protein embeddings. This work advances our ability to model mutation-induced perturbations in protein-protein interactions and opens new avenues for structure-aware machine learning in protein engineering.

## Keywords

Protein-protein interactions, Binding free energy change (ΔΔG), Deep learning, Interpretable models, Multi-level features, Sequence-structure dependencies, Computational biology