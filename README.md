# The IFS-Former

A Transformer-based model for trip energy consumption prediction of battery electric vehicles under inconsistent feature spaces.

### 1. Installation
1. Create the environment.
```bash
conda create -n bev_energy python=3.10
```
2. Install the requirements.
```bash
conda activate bev_energy
pip install -r requirements.txt
```

### 2. Usage
1. Decompress the rar file in ```./checkpoint``` and get the pre-trained pth file.
2. Run few-shot learning.
```bash
python few_shot_learning.py
```
