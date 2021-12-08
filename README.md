# BanditMaterialsExplorer
Pytorch implementation of Bandit Optimization algorithms for materials exploration. <br>
Details can be found in the paper to be submitted to NeurIPS 2021 Workshop AI for Science
(https://openreview.net/forum?id=xJhjehqjQeB)<br>

# Features
### model
Two-stage learning.
1. Self-Supervised Representation Learning using CGCNN (https://github.com/txie-93/cgcnn)
2. Bandit Optimization using Thompson Sampling

### Problem to be solved
*Explore the materials as close to 2.534 eV indirect bandgap as possible from 6,218 candidates with 100 prior information.*

### The best model explored more efficiently than over 80% of skilled experts/scientists.
Solid lines and painted regions represent mean values and 95% confidence intervals, respectively.
Prior work based on continuous Bayesian optimization cannot explore better materials at the 30th trial than human experts.
Ours won 80% of human experts.<br>
![GitHub Logo](./results/efficiency.png)

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
Open **human_experiment.ipynb**, then run each cell.<br>
Explore materials exploiting information, such as lattice parameters, chemical formula, electronegativity.
Do it 30 times within 60 mins, then generated **trial_result.csv** is your result.
You can compare your exploration efficiency with the model.

### 2. Bandit optimization
Run **run_bandit.py** <br>
This model will explore the 6,218 candidates using 12-dim descriptors extracted through CGCNN.
The 12-dim descriptors are already converted and saved in **descriptors.csv** in the database folder.
Therefore, you don't need to run self-supervised representation learning.

### 3. self-supervised learning using CGCNN
If you want to convert cif files into descriptors, you can use pretraied model. <br>
1. go into **cgcnn** directory
2. unzip **cif.zip**
3. run **extract.py** to convert cif files into 128-dim descriptors using pretrained model (model_best.pth.tar)
4. run **pca.py** to reduce 128-dim into 12 dim descriptors.

### Citation
Masaki Adachi, High-Dimensional Discrete Bayesian Optimization with Self-Supervised Representation Learning for Data-Efficient Materials Exploration, NeurIPS Workshop AI for Science 2021
