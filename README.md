![image](./data/overview.png)

# IGMI

# Contents

- [Overview](#overview)
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Usage](#Usage)
- [Contact](#Contact)


# Overview

**IGMI** is a novel interpretable graph-based multi-level feature interaction model for mutation impact prediction. IGMI integrates multi-level features, including the 1D sequence distance map layer, 3D structure distance map layer, 3D coordinate layer, and atomic-based side-chain geometric features, to effectively capture the inherent complex dependencies in PPI data, thereby improving the accuracy of ∆∆G predictions. The core of the IGMI architecture lies in two variant network modules based on attention mechanisms. These modules are designed to model the interdependencies and information flow across multidimensional features, including 1D sequence distance maps, 3D structural distance maps, and 3D coordinate features, as well as multi-scale features such as residue- and atomic-based information. By integrating these multidimensional and multi-scale features, IGMI effectively models the intricate relationships between sequence and structural characteristics.

# Requirements

## Hardware Requirements

IGMI has been tested on a standard computer equipped with an Intel Core i9 processor, NVIDIA GeForce RTX 4090 GPU and 128 GB of RAM.


## Software Requirements

IGMI supports Linux. It has been tested on Ubuntu 22.04.

# Installation

The model was tested on Linux using  `Python 3.8`, `PyTorch 2.7.0` ,`easydict 1.13` ,`numpy 2.2.5` ,`pandas 2.2.3`, and `Biopython 1.7.1`. The dependencies can be set up using the following commands:

```
git clone https://github.com/ShiweiWu-545/IGMI.git
cd IGMI

conda create --name IGMI python=3.8
conda activate IGMI
pip install -r requirements.txt
```
The IGMI dependency configuration process takes about 2 hours on a standard computer.

# Usage

The model requires two input PDB files: (1) a wild-type complex structure, and (2) a mutated complex structure. The mutated structures are typically built by protein design packages such as [Rosetta](https://www.rosettacommons.org/docs/latest/cartesian-ddG). Note that both structures must have the same length. The DDG can be predicted for the two structures by running the command:

```
python ./run/run_model.py
```
Outputs
```
DDG: tensor([0.8404], device='cuda:0', grad_fn=<SumBackward1>)
Positive values indicate a decrease in affinity and negative values indicate an increase in affinity.
```

# Contact

Please contact wushiwei@hrbeu.edu.cn for any questions related to the source code.
