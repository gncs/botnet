# BOTNet

Implementation of a set of interatomic potentials in the design space of equivariant message passing. (very badly said).


## Installation

To install the dependencies on cpu run following line,

```bash
sbatch ./scripts/setup_env.sh
```
For cuda depdencies, edit the second line of the file from "CUDA"=cpu to the prefered version of cuda (eg. cu102).

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

#### BOTNet Scale Shifted
#### BOTNet E0
#### NequIP 
#### NequIP Linear


## Figures

## Contacts

Ilyes Batatia: ilyes.batatia@ens-paris-saclay.fr 

Gregor Simm: 

