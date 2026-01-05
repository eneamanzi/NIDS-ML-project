# NIDS-ML-project

python3 -m venv .venv

source .venv/bin/activate

deactivate


pip install -r requirements.txt



#!/bin/bash
curl -L -o ~/Downloads/ciciot2023.zip\
  https://www.kaggle.com/api/v1/datasets/download/himadri07/ciciot2023



# 1. Setup dataset (scarica da Kaggle)
python setup_dataset.py

# 2. Preprocessing (usa sample per test veloce)
python src/preprocessing_ciciot2023.py --sample 10000

# 3. Train Decision Tree
python src/train_decision_tree_ciciot2023.py

# 4. Train Random Forest (modello principale)
python src/train_random_forest_ciciot2023.py




