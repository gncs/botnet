# BOTNet
This repository provides the code for the paper titled *"The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials".*
> [!CAUTION]
> This repository is intended exclusively for reproducing the results presented in the paper. The code is not optimized for general use.
## Table of Contents
- [BOTNet](#botnet)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Datasets](#datasets)
  - [Experiments](#experiments)
    - [Scale-Shifted BOTNet](#scale-shifted-botnet)
    - [BOTNet E0](#botnet-e0)
    - [NequIP](#nequip)
  - [Reference](#reference)
  - [Contacts](#contacts)
## Installation
1. Install PyTorch. Please refer to the [official PyTorch documentation](https://pytorch.org/get-started/locally/) for installation instructions.
2. Install additional dependencies
   ```bash
   pip install pip --upgrade
   pip install e3nn==0.4.4 opt_einsum ase==3.22.1 torch_ema prettytable
   ```
3. Finally, install the BOTNet package:
   ```bash
   git clone https://github.com/gncs/botnet
   pip install botnet/
   ```
## Datasets
The datasets required for reproducing the 3BPA, ethanol, and AcAc experiments can be downloaded from this repository: [https://github.com/davkovacs/BOTNet-datasets](https://github.com/davkovacs/BOTNet-datasets). Please download the relevant tar.gz files and place them in a folder named `downloads` within the same directory as the code.

For the rMD17 and iso17 experiments, the data will be downloaded automatically.

## Experiments
The results presented in the paper can be reproduced from the command line.
For instance, to train a "scale-shifted BOTNet" on 3BPA configurations at 300K run the following command:
```bash
python3 ./botnet-main/scripts/run_train.py \
    --dataset="3bpa" \
    --subset="train_300K" \
    --seed=2 \
    --model="scale_shift_non_linear" \
    --device=cuda \
    --max_num_epochs=3000 \
    --patience=256 \
    --name="Botnet_3BPA" \
    --energy_weight=27.0 \
    --forces_weight=729.0 \
    --hidden_irreps='80x0o + 80x0e + 80x1o + 80x1e + 80x2o + 80x2e + 80x3o + 80x3e' \
    --batch_size=5 \
    --interaction_first="AgnosticNonlinearInteractionBlock" \
    --interaction="AgnosticResidualNonlinearInteractionBlock" \
    --ema \
    --ema_decay=0.99 \
    --scaling='rms_forces_scaling' \
    --weight_decay=0.0 \
    --restart_latest
```

To train on other experiments, please change `--dataset` (see options in the argparser), and change the `--subset` accordingly. For datasets with different splits, one can specify a `--split` argument. 
For example to train on the first ethanol split of rMD17, use,
```bash
--dataset="rmd17" \
--subset="ethanol" \
--split=1 \
```

To train specific models, select the command line options in the subsections below.
### Scale-Shifted BOTNet
- **Agnostic Scale Shifted BOTNet** (no chemical dependency in the radial basis):
   ```bash
   --model="scale_shift_non_linear" \
   --interaction_first="AgnosticNonlinearInteractionBlock" \
   --interaction="AgnosticResidualNonlinearInteractionBlock" \
   ```
- **Element Dependent Scale Shifted BOTNet** (chemical dependency in the radial basis)
   ```bash
   --model="scale_shift_non_linear" \
   --interaction_first="AgnosticNonlinearInteractionBlock" \
   --interaction="ResidualElementDependentInteractionBlock" \
   ```
- **Fully Residual Element Dependent Scale Shifted BOTNet** (chemical dependency in the radial basis and residual connection even at the first layer)
   ```bash
   --model="scale_shift_non_linear" \
   --interaction_first="ResidualElementDependentInteractionBlock" \
   --interaction="ResidualElementDependentInteractionBlock" \
   ```
### BOTNet E0
-  **Agnostic BOTNet** (no chemical dependency in the radial basis and residual even at the first layer)
   ```bash
   --model="body_ordered_non_linear" \
   --interaction_first="AgnosticNonlinearInteractionBlock" \
   --interaction="AgnosticResidualNonlinearInteractionBlock" \
   ```
- **Element Dependent BOTNet** (chemical dependency in the radial basis and residual even at the first layer)
   ```bash
   --model="body_ordered_non_linear" \
   --interaction_first="AgnosticNonlinearInteractionBlock" \
   --interaction="ResidualElementDependentInteractionBlock" \
   ```
### NequIP
- **NequIP**
   ```bash
   --model="scale_shift_non_linear_single_readout" \
   --interaction_first="NequIPInteractionBlock" \
   --interaction="NequIPInteractionBlock" \
   --gate="None" \
   ```
- **NequIP linear** (no non-linearities except in the radial basis)
   ```bash
   --model="scale_shift_non_linear_single_readout" \
   --interaction_first="AgnosticResidualNonlinearInteractionBlock" \
   --interaction="AgnosticResidualNonlinearInteractionBlock" \
   --gate="None" \
   ```
## Reference
```bibtex
@misc{batatia2022designspace,
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
Ilyes Batatia (`ib467 at cam.ac.uk`) and Gregor Simm (`gncsimm at gmail.com`)
