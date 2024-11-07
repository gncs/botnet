
# BOTNet

This repository provides the BOTNet code used in the paper: *"The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials."* **This repository is intended solely for reproducing the results presented in the paper.**  
Refer to the [Experiments](#experiments) section for instructions on defining and running the different models used in the study.


## Table of Contents 
1. [Installation](#installation) 
2. [Experiments](#experiments) 
	- [BOTNet Scale Shifted](#botnet-scale-shifted) 
	- [BOTNet E0](#botnet-e0) 
	- [NequIP](#nequip)
	- [NequIP Linear](#nequip-linear) 
3. [References](#references) 
4. [Contacts](#contacts)

## Installation

### 1. Install Pytorch
To install the package, **make sure to install PyTorch.** Please refer to the [official PyTorch installation](https://pytorch.org/get-started/locally/) for the installation instructions. Select the appropriate options for your system.

### 2. Install dependencies

```bash
pip install pip --upgrade
pip install e3nn==0.4.4 opt_einsum ase torch_ema prettytable
git clone https://github.com/gncs/botnet
pip install botnet/
```

## Experiments

All the experiments of the paper can be reproduced from the command line using the parser script. We first give a detail example on how to train a scale shifted BOTNet on 3BPA configurations at 300K :

```bash
# Run command
python3 ./botnet-main/scripts/run_train.py \
    --dataset="3bpa" \    #Select the dataset (choices 3bpa, acac, ethanol, iso17, md17)
    --subset="train_300K" \   #Select the training file {subset}.xyz
    --seed=2 \  
    --model="scale_shift_non_linear" \ #Select the type of model to use
    --device=cuda \
    --max_num_epochs=3000 \
    --patience=256 \  #Number of increasing loss before stopping training
    --name="Botnet_3BPA" \ 
    --energy_weight=27.0 \ #Weight of the energy in the loss (recommanded equal to the average number of atoms in the training set)
    --forces_weight=729.0 \ #Weight of the forces in the loss (recommanded to the square of the number of atoms in the training set).
    --hidden_irreps='80x0o + 80x0e + 80x1o + 80x1e + 80x2o + 80x2e + 80x3o + 80x3e' \ #The irreducible representations of hidden features in the network
    --batch_size=5 \
    --interaction_first="AgnosticNonlinearInteractionBlock" \ #The type of interaction to use at the first layer
    --interaction="AgnosticResidualNonlinearInteractionBlock" \ #The type of interaction to use for all the subsequent layers
    --ema \ # Enables the exponential moving average
    --ema_decay=0.99 \ # Exponential moving average decay rate
    --scaling='rms_forces_scaling' \ #Type of normalization to apply
    --weight_decay=0.0 \ #Weight decay, recommanded to keep at 0
    --restart_latest \
```

### BOTNet Scale Shifted
To run **Agnostic Scale Shifted BOTNet** (meaning **no** chemical dependency in the radial basis) select,

```bash
    --model="scale_shift_non_linear" #Select the type of model to use
    --interaction_first="AgnosticNonlinearInteractionBlock" \ #The type of interaction to use at the first layer
    --interaction="AgnosticResidualNonlinearInteractionBlock" \ #The type of interaction to use for all the subsequent layers
```

To run **Element Dependent Scale Shifted BOTNet** (meaning chemical dependency in the radial basis) select,

```bash
    --model="scale_shift_non_linear" #Select the type of model to use
    --interaction_first="AgnosticNonlinearInteractionBlock" \ #The type of interaction to use at the first layer
    --interaction="ResidualElementDependentInteractionBlock" \ #The type of interaction to use for all the subsequent layers
```

To run **Fully Residual Element Dependent Scale Shifted BOTNet** (meaning chemical dependency in the radial basis and residual even at the first layer) select,

```bash
    --model="scale_shift_non_linear" #Select the type of model to use
    --interaction_first="ResidualElementDependentInteractionBlock" \ #The type of interaction to use at the first layer
    --interaction="ResidualElementDependentInteractionBlock" \ #The type of interaction to use for all the subsequent layers
```

### BOTNet E0

To run **Agnostic BOTNet** (meaning **no** chemical dependency in the radial basis and residual even at the first layer) select,

```bash
    --model="body_ordered_non_linear" #Select the type of model to use
    --interaction_first="AgnosticNonlinearInteractionBlock" \ #The type of interaction to use at the first layer
    --interaction="AgnosticResidualNonlinearInteractionBlock" \ #The type of interaction to use for all the subsequent layers
```

To run **Element Depedent BOTNet** (meaning chemical dependency in the radial basis and residual even at the first layer) select,

```bash
    --model="body_ordered_non_linear" #Select the type of model to use
    --interaction_first="AgnosticNonlinearInteractionBlock" \ #The type of interaction to use at the first layer
    --interaction="ResidualElementDependentInteractionBlock" \ #The type of interaction to use for all the subsequent layers
```


### NequIP 

To run **NequIP** select,

```bash
    --model="scale_shift_non_linear_single_readout" #Select the type of model to use
    --interaction_first="NequIPInteractionBlock" \ #The type of interaction to use at the first layer
    --interaction="NequIPInteractionBlock" \ #The type of interaction to use for all the subsequent layers
    --gate="None" \
```

### NequIP Linear

To run **NequIP Linear** (meaning no non linearities except in the radial basis) select,

```bash
    --model="scale_shift_non_linear_single_readout" #Select the type of model to use
    --interaction_first="AgnosticResidualNonlinearInteractionBlock" \ #The type of interaction to use at the first layer
    --interaction="AgnosticResidualNonlinearInteractionBlock" \ #The type of interaction to use for all the subsequent layers
    --gate="None" \
```

## References

```bibtex
@misc{batatia2022designspacee3equivariantatomcentered,
      title={The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials}, 
      author={Ilyes Batatia and Simon Batzner and Dávid Péter Kovács and Albert Musaelian and Gregor N. C. Simm and Ralf Drautz and Christoph Ortner and Boris Kozinsky and Gábor Csányi},
      year={2022},
      eprint={2205.06643},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2205.06643}, 
}
```

## Contacts

Ilyes Batatia: ib467@cam.ac.uk

Gregor Simm: gncsimm@gmail.com

