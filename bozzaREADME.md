# NIDS-ML-Project - CICIoT2023

Network Intrusion Detection System usando Machine Learning con il dataset **CICIoT2023**.

## 📊 Dataset: CICIoT2023

- **Source**: Canadian Institute for Cybersecurity (CIC)
- **Anno**: 2023
- **Tipo**: IoT Network Traffic
- **Features**: 46 features estratte da flussi di rete
- **Attacchi**: 33 tipi divisi in 7 categorie
  - DoS (Denial of Service)
  - DDoS (Distributed DoS)
  - Recon (Reconnaissance/Port Scan)
  - Web-based
  - Brute Force
  - Spoofing
  - Mirai
- **Dimensione**: ~46 milioni di record (~13GB non compresso)
- **Download**:
  - https://www.unb.ca/cic/datasets/iotdataset-2023.html
  - https://www.kaggle.com/datasets/himadri07/ciciot2023

## 🎯 Obiettivi Progetto

Sviluppare un NIDS che:
- Classifica traffico di rete in **Normal** (0) vs **Anomaly** (1)
- Raggiunge:
  - **Accuracy** ≥ 95%
  - **Precision** ≥ 90%
  - **Recall** ≥ 95%

## 🚀 Setup

### 1. Crea ambiente virtuale

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows
```

### 2. Installa dipendenze

```bash
pip install -r requirements.txt
```

### 3. Scarica il dataset

Opzione A - Kaggle (raccomandato):
```bash
# Installa Kaggle CLI
pip install kaggle

# Configura credenziali (vedi https://www.kaggle.com/docs/api)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Scarica dataset
kaggle datasets download -d himadri07/ciciot2023
unzip ciciot2023.zip -d data/raw/CICIoT2023/
```

Opzione B - Download manuale:
1. Vai su https://www.kaggle.com/datasets/himadri07/ciciot2023
2. Scarica il dataset
3. Estrai i file CSV in `data/raw/CICIoT2023/`

### 4. Struttura directory

```
NIDS-ML-project/
├── data/
│   ├── raw/
│   │   └── CICIoT2023/        # File CSV del dataset qui
│   │       ├── part-00000-*.csv
│   │       ├── part-00001-*.csv
│   │       └── ...
│   └── processed/              # Dati preprocessati (generati automaticamente)
├── notebooks/
│   └── 01_data_exploration_ciciot2023.ipynb
├── src/
│   ├── preprocessing_ciciot2023.py
│   ├── train_decision_tree_ciciot2023.py
│   └── train_random_forest_ciciot2023.py
├── models/                     # Modelli salvati
├── docs/                       # Grafici e report
└── requirements.txt
```

## 📝 Workflow

### Step 1: Data Exploration

Esplora il dataset con Jupyter:

```bash
jupyter notebook notebooks/01_data_exploration_ciciot2023.ipynb
```

### Step 2: Preprocessing

**⚠️ IMPORTANTE**: Il dataset completo è molto grande (~46M record). Per un primo test, usa il parametro `--sample`:

```bash
# Test con sample (10k righe per file)
python src/preprocessing_ciciot2023.py --sample 10000

# Oppure usa il dataset completo (richiede molto tempo e RAM)
python src/preprocessing_ciciot2023.py
```

Output:
- `data/processed/X_train.pkl` - Features training (normalizzate)
- `data/processed/X_test.pkl` - Features test (normalizzate)
- `data/processed/y_train.pkl` - Labels training (0=Normal, 1=Anomaly)
- `data/processed/y_test.pkl` - Labels test
- `data/processed/feature_names.pkl` - Nomi features
- `data/processed/scaler.pkl` - StandardScaler fitted
- `data/processed/dataset_info.pkl` - Info dataset

### Step 3: Train Decision Tree (Baseline)

```bash
python src/train_decision_tree_ciciot2023.py
```

Output:
- `models/DecisionTree_CICIoT2023.pkl` - Modello salvato
- `docs/confusion_matrix_dt_ciciot2023.png` - Confusion matrix
- `docs/roc_curve_dt_ciciot2023.png` - ROC curve
- `docs/feature_importance_dt_ciciot2023.png` - Feature importance

### Step 4: Train Random Forest (Modello Principale)

```bash
python src/train_random_forest_ciciot2023.py
```

Output:
- `models/RandomForest_CICIoT2023.pkl` - Modello salvato
- `models/best_model_ciciot2023.pkl` - Copia del miglior modello
- `docs/confusion_matrix_rf_ciciot2023.png` - Confusion matrix
- `docs/roc_curve_rf_ciciot2023.png` - ROC curve
- `docs/feature_importance_rf_ciciot2023.png` - Feature importance
- `docs/metrics_comparison_rf_ciciot2023.png` - Confronto metriche

## 🎛️ Parametri Personalizzabili

### Preprocessing

```bash
python src/preprocessing_ciciot2023.py \
    --data-dir data/raw/CICIoT2023 \
    --output-dir data/processed \
    --test-size 0.2 \
    --sample 50000 \
    --seed 42
```

### Decision Tree

```bash
python src/train_decision_tree_ciciot2023.py \
    --data-dir data/processed \
    --output-dir docs \
    --max-depth 15 \
    --model-path models/DecisionTree_CICIoT2023.pkl
```

### Random Forest

```bash
python src/train_random_forest_ciciot2023.py \
    --data-dir data/processed \
    --output-dir docs \
    --n-estimators 100 \
    --max-depth 20 \
    --model-path models/RandomForest_CICIoT2023.pkl
```

## 📈 Metriche Attese

Con il dataset CICIoT2023 e Random Forest ottimizzato:

| Metrica | Target | Tipico |
|---------|--------|--------|
| Accuracy | ≥0.95 | 0.96-0.99 |
| Precision | ≥0.90 | 0.92-0.98 |
| Recall | ≥0.95 | 0.96-0.99 |
| F1-Score | - | 0.94-0.98 |
| AUC-ROC | - | 0.97-0.99 |

**Note**:
- **Recall** è la metrica più critica per NIDS (minimizzare falsi negativi)
- **False Negative Rate** deve essere <5% (attacchi non rilevati)
- **False Positive Rate** dovrebbe essere <10% (allarmi falsi)

## 🔍 Feature Engineering

Le 46 features del CICIoT2023 includono:
- Flow duration
- Packet counts (forward/backward)
- Byte counts (forward/backward)
- Inter-arrival times (mean, std, min, max)
- Packet length statistics
- Flow rate
- Flags (FIN, SYN, RST, PSH, ACK, URG)
- Header length
- Payload statistics

Le features più importanti tipicamente sono:
1. Flow duration
2. Packet counts
3. Byte transfer rates
4. Inter-arrival time statistics
5. Packet length statistics

## 💡 Tips & Troubleshooting

### Dataset troppo grande

Se hai problemi di RAM:

```bash
# Usa sample più piccolo
python src/preprocessing_ciciot2023.py --sample 5000

# Oppure usa solo alcuni file CSV
# Modifica load_ciciot2023() in preprocessing_ciciot2023.py
# per caricare solo i primi N file
```

### Training lento

Per Random Forest:

```bash
# Riduci numero alberi (compromesso speed/accuracy)
python src/train_random_forest_ciciot2023.py --n-estimators 50

# O riduci profondità
python src/train_random_forest_ciciot2023.py --max-depth 15
```

### Accuracy bassa

Se non raggiungi i target:

1. **Aumenta complessità modello**:
   - `--n-estimators 200`
   - `--max-depth 25`

2. **Gestisci class imbalance**:
   - Implementa SMOTE (Synthetic Minority Over-sampling)
   - Usa `class_weight='balanced'` in RandomForest

3. **Feature engineering**:
   - Rimuovi features ridondanti (alta correlazione)
   - Crea nuove features (es. ratio, combinazioni)

4. **Hyperparameter tuning**:
   - GridSearchCV o RandomizedSearchCV
   - Ottimizza `min_samples_split`, `min_samples_leaf`

## 📚 Riferimenti

- **Paper**: Neto et al. (2023). "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment." Sensors.
- **Dataset**: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- **Scikit-learn**: https://scikit-learn.org/

## 👥 Team

- Sviluppatori: 2
- Durata: ~2 mesi
- Corso: Machine Learning per Sicurezza Informatica

## 📄 License

Progetto educativo - Università

---

**Note**: Questo README è specifico per il dataset CICIoT2023. Per dataset NSL-KDD (versione precedente), vedi `README_NSL-KDD.md`.

-  -  -  
**ROBA UTILE**
1. A cosa servono le 3 cartelle (Train, Test, Validation)?

Nel Machine Learning professionale (e su dataset grandi come questo), i dati sono già divisi per evitare errori metodologici gravi. Ecco come dovete usarle:

    train (Allenamento): È il "libro di testo" su cui il modello studia.

        Quando usarla: ORA. Userete questa cartella per l'Esplorazione (EDA), per il Preprocessing e per il model.fit().

        Contenuto: Probabilmente contiene decine (o centinaia) di file CSV. Non è un file unico.

    validation (Validazione): È la "simulazione d'esame".

        Quando usarla: Durante il training, alla fine di ogni "epoca" (o iterazione). Serve per capire se il modello sta imparando bene o se sta imparando a memoria (Overfitting). Serve a calibrare gli iperparametri.

        Contenuto: Dati che il modello non vede durante l'addestramento diretto.

    test (Test Finale): È l' "esame finale".

        Quando usarla: MAI fino all'ultimo giorno. Si usa una volta sola, alla fine del progetto, per calcolare l'accuratezza finale da mettere nel report.

        Regola d'oro: Se fate Data Exploration sulla cartella test e prendete decisioni basandovi su quella, state barando ("Data Leakage"). Non toccatela per ora.