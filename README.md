# BanditMaterialsExplorer
Pytorch implementation of Bandit Optimization algorithms for materials exploration. <br>
Details will be the paper to be submitted to NeurIPS 2021 Workshop AI for Science(https://ai4sciencecommunity.github.io). <br>

# Features
### model
two stage learning.
1. self-supervised representation learning using CGCNN (https://github.com/txie-93/cgcnn)
2. Bandit Optimization using Thompson sampling

### problem to be solved
Explore the materials as close to 2.534 eV indirect bandgap as poosible from 6,218 candidates with 100 prior information.

### The best model explored more efficiently than over 80% of skilled experts/scientists.



# Requirements
- Python 3.6
- PyTorch 1.4
- pymatgen
- scikit-learn
- GPyTorch
- BoTorch
- Ax-platform

# Usage
### 1. human exploration efficiency experiment
Open human_experiment.ipynb, then run each cells.
Explore materials exploiting information, such as lattice parameters, chemical formula, electronegativity.
Do it 30 times within 60 mins, then generated **trial_result.csv** is your result.
You can compare your exploration efficiency with the model.

### 2. Bandit optimization
In sa
```
python3 read_cif.py --input ./cif --output ./lithium_datasets.pkl
```
**lithium_datasets.pkl** will be created.

Second, convert the checked results into XRD spectra database.
```
python3 convertXRDspectra.py --input ./lithium_datasets.pkl --batch 8 --n_aug 5
```
**XRD_epoch5.pkl** will be created.

# Train
```
python3 train_model.py --input ./XRD_epoch5.pkl --output learning_curve.csv --batch 16 --n_epoch 100
```
**Output data**
- Trained model -> **regnet1d_adacos_epoch100.pt**
- Learning curve -> **learning_curve.csv**
- Correspondence between numerical int label and crystal names -> **material_labels.csv**

# Result
- Database: Lithium compounds (8,172)
- XRD: 2 theta 0 - 120 degree with 0.02 width (6,000 dims)
- model: 1D-CNN (1D-RegNet) + Deep metric learning (AdaCos)
- Loss: CrossEntropyLoss
- Metric: Top 5 accuracy (%)
- epoch: 100

| Train         | Validation    | Test  |
| ------------- |:-------------:| -----:|
| 99.41         | 97.30         | 97.30 |

# Citation
### Papers
- AdaCos: https://arxiv.org/abs/1905.00292
- 1D-RegNet: https://arxiv.org/abs/2008.04063
- Physics-informed data augmentation: https://arxiv.org/abs/1811.08425v2

### Implementation
- AdaCos: https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
- 1D-RegNet: https://github.com/hsd1503/resnet1d
- Physics-informed data augmentation: https://github.com/PV-Lab/autoXRD
- Top k accuracy: https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
