# Egret Neural Network Potentials

<p align="center">
<img src="docs/visual_abstract.png" alt="Visual Abstract" width="650"/>
</p>

This repository contains the **Egret** family of neural network potentials, developed by Rowan using the [MACE](https://github.com/ACEsuit/mace) architecture. You can use the pretrained model weights locally or run predictions directly via the [Rowan web platform](https://labs.rowansci.com/).

For more information on the models, see [the preprint](https://rowansci.com/publications/egret-1-pretrained-neural-network-potentials).

For questions or issues, please open a GitHub issue or contact the Rowan team at contact@rowansci.com.

## Egret Model Suite

We provide three general-purpose models, released under the MIT license:

- **Egret-1** — optimized for bioorganic molecules; a strong general-purpose model  
- **Egret-1e** — enhanced with main-group chemistry data; excels at thermochemistry  
- **Egret-1t** — trained on transition states; ideal for modeling chemical reactivity

We also provide 2 smaller versions of the Egret-1 model under the same MIT license:
- **Egret-1S** — invariant-only with 92 spherical channels; fastest model with smallest memory footprint
- **Egret-1M** — low-resolution equivariance with 128 spherical channels; fast equivariant model with a smaller memory footprint

## Example: Using Egret-1 with ASE
Egret-1 is compatible with the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) interface. The following is an example using the `mace_off` calculator from the [`mace-torch`](https://github.com/ACEsuit/mace) package:

### Energy and Force Prediction

```python
import ase.io
from ase.calculators.calculator import all_changes

from mace.calculators import mace_off

atoms = read("<path_to_molecule_file>")

calculator = mace_off(model="<path_to_model/EGRET_1.model>", default_dtype="float64")

calculator.calculate(atoms, ["<ASE_task_1>", "<ASE_task_2>"], all_changes)

print(calculator.results)
```

### Extracting Node Features

To get the full descriptors from a model (of shape `[L,1920]` for standard models, `[L,640]` for Egret-1M, and `[L,192]` for Egret-1S, where `L` is the number of atoms):

```python
equivariant_descriptor = calculator.models[0](
    calculator._atoms_to_batch(atoms).to_dict(),
)["node_feats"]

```

To extract the invariant descriptors (Egret-1S descriptors are already invariant):

```python
//standard models, shape = [Lx384]
invariant_descriptor = torch.cat([
    full_descriptor[:, :192],
    full_descriptor[:, -192:],
], dim=1)

//Egret-1M, shape = [Lx256]
invariant_descriptor = torch.cat([
    full_descriptor[:, :128],
    full_descriptor[:, -128:],
], dim=1)

```

Invariant features are recommend for scalar property prediction tasks that do not change with rotation like energy or HOMO-LUMO gap. Equivariant features are recommended for tasks that do depend on rotation like dipole moments or forces.


## Local Usage
Install the required packages using 
```
conda env create -f environment.yml
conda activate egret-env      
```

To run the example script, run 
```
python example.py
```

These models can run on either CPU or GPU. 

