# NIDS CICIoT2023 - Multi-Class Classification

Network Intrusion Detection System con Machine Learning su dataset **CICIoT2023**.

## 🎯 Classificazione Multi-Classe

Il sistema classifica il traffico di rete in **8 macro-categorie**:

1. **Benign** - Traffico normale
2. **DDoS** - Distributed Denial of Service (12 varianti)
3. **DoS** - Denial of Service (4 varianti)
4. **Mirai** - Botnet Mirai (3 varianti)
5. **Recon** - Reconnaissance/Port Scanning (5 varianti)
6. **Web** - Web attacks: SQLi, XSS, Command Injection (5 varianti)
7. **Spoofing** - ARP/DNS Spoofing (2 varianti)
8. **BruteForce** - Dictionary attacks (1 variante)

**Totale**: 34 attacchi specifici → 8 macro-categorie

## 📊 Dataset

- **Nome**: CICIoT2023
- **Source**: Canadian Institute for Cybersecurity
- **Anno**: 2023
- **Features**: 46 features estratte da traffico IoT
- **Records**: ~46 milioni (dataset completo)
- **Format**: CSV (train.csv, test.csv, validation.csv)
- **Download**: https://www.kaggle.com/datasets/himadri07/ciciot2023

### ⚠️ Problema Sbilanciamento

Il dataset presenta **forte sbilanciamento**:
- ~97% attacchi
- ~3% traffico benign

**Soluzione implementata**: `class_weight='balanced'` in Decision Tree e Random Forest

## 🚀 Workflow Completo

### Step 1: Setup Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Step 2: Download Dataset

**Da Kaggle** (raccomandato):
```bash
# Download da Kaggle CLI
curl -L -o ~/Downloads/ciciot2023.zip\
  https://www.kaggle.com/api/v1/datasets/download/himadri07/ciciot2023
unzip ciciot2023.zip -d data/raw/CICIOT23/

# Il dataset dovrebbe avere questa struttura:
data/raw/CICIOT23/
├── train/train.csv
├── test/test.csv
└── validation/validation.csv
```

### Step 3: Preprocessing

**Questo script esegue**:
1. Caricamento train/test/val
2. Pulizia dati (missing, infiniti, duplicati)
3. Mapping 34 classi → 8 macro-categorie
4. Creazione doppia label: `y_macro` (training) + `y_specific` (logging)
5. Anti-leakage: fit scaler/encoder solo su train
6. Normalizzazione features (StandardScaler)
7. Salvataggio in formato PKL (veloce + compatto)

**Comando**:
```bash
# Test con 10k righe (veloce)
cd src
python3 ./preprocessing.py --nrows 10000
```

Oppure 

```bash
# Dataset completo (richiede tempo e RAM)
python src/data_processing.py \
    --train-path data/raw/CICIOT23/train/train.csv \
    --test-path data/raw/CICIOT23/test/test.csv \
    --val-path data/raw/CICIOT23/validation/validation.csv \
    --output-dir data/processed/CICIOT23 \
    --nrows 10000
```


**Output** (formato PKL):
```
data/processed/CICIOT23/
├── train_processed.pkl       # Features + y_macro_encoded + y_specific
├── test_processed.pkl        # Features + y_macro_encoded + y_specific
├── val_processed.pkl         # Features + y_macro_encoded + y_specific
├── scaler.pkl                # StandardScaler (per produzione)
├── label_encoder.pkl         # LabelEncoder (per produzione)
└── mapping_info.json         # Info mapping + metadata
```

### Step 4: Train Decision Tree (Baseline)

```bash
cd ./src
python3 ./train_decision_tree.py
    #--max-depth 20
```

**Output**:
- `models/DecisionTree_CICIoT2023_multiclass.pkl`
- `docs/decision_tree/confusion_matrix_dt_multiclass.png`
- `docs/decision_tree/per_class_performance_dt_multiclass.png`
- `docs/decision_tree/feature_importance_dt_multiclass.png`

### Step 5: Train Random Forest (Modello Principale)

```bash
python src/train_random_forest_multiclass.py \
    --data-dir data/processed/CICIOT23 \
    --output-dir docs \
    --n-estimators 100 \
    --max-depth 25
```

**Output**:
- `models/RandomForest_CICIoT2023_multiclass.pkl`
- `models/best_model_ciciot2023_multiclass.pkl` (copia miglior modello)
- `docs/confusion_matrix_rf_multiclass.png`
- `docs/per_class_performance_rf_multiclass.png`
- `docs/feature_importance_rf_multiclass.png`
- `docs/metrics_comparison_rf_multiclass.png`

## 📈 Metriche Target

| Metrica | Target | Note |
|---------|--------|------|
| **Accuracy** | ≥ 0.95 | Obiettivo principale |
| **Precision** | ≥ 0.90 | Minimizzare falsi positivi |
| **Recall** | ≥ 0.95 | **CRITICO**: minimizzare falsi negativi |
| **F1-Score** | - | Bilanciamento precision/recall |

### 🎯 Interpretazione Metriche Multi-Class

**Weighted Average**: Le metriche sono calcolate con media pesata per classe, considerando lo sbilanciamento.

**Per-Class Metrics**: Ogni macro-categoria ha le sue metriche separate, utili per identificare:
- Classi ben classificate (Precision/Recall alti)
- Classi problematiche (Precision/Recall bassi)

## 🔧 Parametri Configurabili

### Preprocessing

```bash
python src/data_processing.py \
    --train-path <path> \
    --test-path <path> \
    --val-path <path> \
    --output-dir data/processed \
    --nrows 50000 \        # Sample size (per testing)
    --label-col label      # Nome colonna label
```

### Decision Tree

```bash
python src/train_decision_tree_multiclass.py \
    --data-dir data/processed/CICIOT23 \
    --output-dir docs \
    --max-depth 20 \       # Profondità albero
    --model-path models/dt.pkl
```

### Random Forest

```bash
python src/train_random_forest_multiclass.py \
    --data-dir data/processed/CICIOT23 \
    --output-dir docs \
    --n-estimators 100 \   # Numero alberi
    --max-depth 25 \       # Profondità alberi
    --model-path models/rf.pkl
```

## 🏗️ Architettura Sistema

### 1. Preprocessing (`data_processing.py`)

```
Raw CSV (34 classi)
    ↓
Cleaning (missing, inf, duplicates)
    ↓
Mapping → 8 macro-categorie
    ↓
Labels: y_macro (0-7) + y_specific (0-33)
    ↓
Normalization (StandardScaler, FIT solo su train)
    ↓
Encoding (LabelEncoder, FIT solo su train)
    ↓
PKL files (train, test, val)
```

### 2. Training (`train_*.py`)

```
Load PKL files
    ↓
Split: X (features) + y_macro (labels)
    ↓
Train Model (class_weight='balanced')
    ↓
Evaluate (Train/Test/Val)
    ↓
Visualizations + Model saving
```

### 3. Doppia Label Strategy

**y_macro_encoded** (0-7):
- Usata per **training** del modello
- 8 classi bilanciate con class_weight
- Permette classificazione robusta

**y_specific** (0-33):
- Mantiene **info dettagliata** attacco originale
- Usata per **logging** e **dashboard**
- Utile per analisi forensi

Esempio:
```
Attacco originale: "DDoS-ICMP_Flood"
→ y_macro: 2 (DDoS)
→ y_specific: 5 (indice specifico per logging)
```

## 🔍 Features Importanti

Le features più rilevanti tipicamente sono:

1. **Duration** - Durata flusso
2. **rst_count** - Count pacchetti RST
3. **ack_flag_number** - Flag ACK
4. **Header_Length** - Lunghezza header
5. **urg_count** - Count pacchetti URG

Vedi `feature_importance_*.png` per l'analisi completa.

## 💡 Tips & Troubleshooting

### Dataset troppo grande

```bash
# Usa sample per test veloce
python src/data_processing.py --nrows 10000
```

### Accuracy bassa

1. **Aumenta complessità**:
   ```bash
   python src/train_random_forest_multiclass.py --n-estimators 200 --max-depth 30
   ```

2. **Implementa SMOTE** (future work):
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
   ```

3. **Feature engineering**:
   - Rimuovi features ridondanti (Rate/Srate)
   - Crea nuove features (ratio, combinazioni)

### Training lento

```bash
# Riduci alberi (compromesso speed/accuracy)
python src/train_random_forest_multiclass.py --n-estimators 50
```

## 📂 Struttura Progetto

```
NIDS-CICIoT2023/
├── data/
│   ├── raw/
│   │   └── CICIOT23/
│   │       ├── train/train.csv          # Dataset originale
│   │       ├── test/test.csv
│   │       └── validation/validation.csv
│   └── processed/
│       └── CICIOT23/
│           ├── train_processed.pkl      # Dataset processato
│           ├── test_processed.pkl
│           ├── val_processed.pkl
│           ├── scaler.pkl               # Artifacts
│           ├── label_encoder.pkl
│           └── mapping_info.json
├── src/
│   ├── data_processing.py               # ⭐ Preprocessing principale
│   ├── train_decision_tree_multiclass.py
│   └── train_random_forest_multiclass.py
├── models/                               # Modelli salvati
├── docs/                                 # Grafici e report
└── notebooks/
    └── 01_data_exploration_ciciot2023.ipynb
```

## 🔐 Mapping Attacchi → Macro-Categorie

### DDoS (12 varianti)
- DDoS-ICMP_Flood
- DDoS-UDP_Flood
- DDoS-TCP_Flood
- DDoS-PSHACK_Flood
- DDoS-RSTFINFlood
- DDoS-SYN_Flood
- DDoS-SynonymousIP_Flood
- DDoS-ICMP_Fragmentation
- DDoS-UDP_Fragmentation
- DDoS-ACK_Fragmentation
- DDoS-HTTP_Flood
- DDoS-SlowLoris

### DoS (4 varianti)
- DoS-UDP_Flood
- DoS-TCP_Flood
- DoS-SYN_Flood
- DoS-HTTP_Flood

### Mirai (3 varianti)
- Mirai-greeth_flood
- Mirai-udpplain
- Mirai-greip_flood

### Spoofing (2 varianti)
- MITM-ArpSpoofing
- DNS_Spoofing

### Recon (5 varianti)
- Recon-HostDiscovery
- Recon-OSScan
- Recon-PortScan
- Recon-PingSweep
- VulnerabilityScan

### Web (5 varianti)
- SqlInjection
- XSS
- CommandInjection
- Uploading_Attack
- BrowserHijacking

### BruteForce (1 variante)
- DictionaryBruteForce

### Backdoor (1 variante)
- Backdoor_Malware

### Benign
- BenignTraffic

## 📚 Riferimenti

- **Dataset**: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- **Paper**: Neto et al. (2023). "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment."
- **Scikit-learn**: https://scikit-learn.org/

## 👥 Team

- Sviluppatori: 2
- Durata: ~2 mesi
- Corso: Machine Learning per Sicurezza Informatica

---

**Note**: Sistema aggiornato per classificazione multi-classe con gestione anti-leakage e doppia label strategy.
