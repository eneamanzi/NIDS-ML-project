# 🛡️ NIDS IoT: Network Intrusion Detection System con Machine Learning

Un sistema avanzato di rilevamento intrusioni (NIDS) progettato specificamente per ambienti IoT, basato sul dataset **CICIoT2023**. Il progetto implementa una pipeline completa di Machine Learning per la classificazione multi-classe di attacchi informatici, confrontando diversi algoritmi e fornendo una dashboard di monitoraggio in tempo reale.

## 🌟 Caratteristiche Principali

* **Classificazione Multi-Classe:** Distingue tra traffico benigno e 7 macro-categorie di attacco (DDoS, DoS, Mirai, Recon, ecc.).
* **Strategia "Double Label":**
* `y_macro`: Usata per il training del modello (8 classi bilanciate).
* `y_specific`: Mantenuta per il logging dettagliato (33 varianti di attacco originali).


* **Pipeline Anti-Leakage:** Preprocessing rigoroso con fitting di scaler ed encoder solo sul training set.
* **Benchmark Comparativo:** Analisi approfondita di 4 algoritmi (DT, RF, k-NN, SVM) con metriche di accuratezza vs latenza.
* **Dashboard Web:** Interfaccia Flask per il monitoraggio delle anomalie e la gestione (simulata) del firewall.

---

## 📊 Performance e Benchmark

Basato sui risultati ottenuti nel file `docs/benchmark_results.csv`, ecco il confronto tra i modelli addestrati sul test set:

| Modello | Accuracy | F1-Score | Latency (ms) | Throughput (pps) | Note |
| --- | --- | --- | --- | --- | --- |
| **Random Forest** | **0.9862** | **0.9865** | 13.73 | ~72 | 🏆 **Miglior Qualità** |
| **Decision Tree** | 0.9765 | 0.9784 | **0.054** | **~18,436** | ⚡ **Miglior Velocità** |
| **k-NN** | 0.9147 | 0.9120 | 0.381 | ~2,625 | Performance medie |
| **SVM** | 0.6850 | 0.7161 | 0.303 | ~3,296 | Non adatto a questo dataset |

> **Analisi:** Il **Random Forest** è il modello più robusto, ideale per analisi offline dove la precisione è critica. Il **Decision Tree**, sebbene leggermente meno preciso, è estremamente veloce (0.05ms per pacchetto), rendendolo l'unica scelta valida per il deployment su dispositivi IoT edge con risorse limitate o per filtraggio in tempo reale ad alto throughput.

---

## 📂 Dataset e Tassonomia Attacchi

Il progetto utilizza il dataset **CICIoT2023**, mappando le 34 classi originali in 8 macro-categorie per semplificare il training senza perdere informazioni critiche.

### Mapping delle Classi

Il sistema converte automaticamente le etichette specifiche nelle seguenti macro-categorie:

1. **Benign**: Traffico normale.
2. **DDoS**: 12 varianti (ICMP Flood, TCP Flood, etc.).
3. **DoS**: 4 varianti (UDP Flood, HTTP Flood, etc.).
4. **Mirai**: 3 varianti di Botnet.
5. **Recon**: 5 varianti (Port Scanning, OS Scan).
6. **Web**: 5 varianti (SQL Injection, XSS).
7. **Spoofing**: ARP e DNS Spoofing.
8. **BruteForce**: Dictionary attacks.

---

## 🚀 Installazione

### Prerequisiti

* Python 3.8+
* Virtual Environment (consigliato)

### Setup Rapido

1. **Clona la repository:**
```bash
git clone https://github.com/tuo-user/NIDS-ML-project.git
cd NIDS-ML-project

```


2. **Crea e attiva l'ambiente virtuale:**
```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows
.venv\Scripts\activate

```


3. **Installa le dipendenze:**
```bash
pip install -r requirements.txt

```



---

## 🛠️ Utilizzo

### 1. Preparazione Dati (Preprocessing)

Scarica il dataset CICIoT2023 e posizionalo in `data/raw/CICIOT23/`. Successivamente, esegui lo script di preprocessing. Questo script gestisce la pulizia, il mapping delle label e la normalizzazione.

```bash
# Esempio rapido (10k righe per test)
python src/preprocessing.py --nrows 10000

# Elaborazione completa
python src/preprocessing.py --train-path data/raw/CICIOT23/train/train.csv

```

*Output:* File `.pkl` ottimizzati in `data/processed/CICIOT23/`.

### 2. Addestramento Modelli

Puoi addestrare singoli modelli o avviare la suite completa.

**Decision Tree (Baseline veloce):**

```bash
python src/train_decision_tree.py

```

**Random Forest (Modello High-Performance):**

```bash
python src/train_random_forest.py --n-estimators 100 --max-depth 25

```

### 3. Confronto Modelli (Benchmark)

Genera il report CSV e i grafici di confronto (inclusi in `docs/`):

```bash
python src/compare_models.py

```

### 4. Avvio Dashboard

La dashboard web permette di visualizzare lo stato del sistema.

```bash
cd dashboard
python app.py

```

Apri il browser su `http://localhost:5000`.

---

## 🏗️ Struttura del Progetto

```text
NIDS-ML-PROJECT/
├── dashboard/               # Web UI (Flask)
│   ├── templates/           # HTML files
│   └── app.py               # Server Dashboard
├── data/
│   ├── raw/                 # CSV originali (non in git)
│   └── processed/           # Dataset .pkl e artefatti (Scaler/Encoder)
├── docs/                    # Grafici e Report generati
│   ├── decision_tree/       # Matrici di confusione DT
│   ├── random_forest/       # Matrici di confusione RF
│   └── benchmark_results.csv # CSV Comparativo
├── models/                  # Modelli addestrati (.pkl)
├── notebooks/               # Jupyter Notebooks per esplorazione dati
├── src/                     # Codice Sorgente
│   ├── compare_models.py    # Script di Benchmark
│   ├── preprocessing.py     # Pipeline di Data Cleaning & Mapping
│   ├── train_decision_tree.py
│   ├── train_knn.py
│   ├── train_random_forest.py
│   └── train_svm.py
├── requirements.txt         # Dipendenze
└── README.md                # Documentazione

```

## 📈 Esempio Visualizzazioni

Il sistema genera automaticamente grafici per analizzare le performance:

* **Accuracy vs Latency Trade-off:** Per scegliere il modello giusto in base ai vincoli hardware.
* **Confusion Matrices:** Per ogni macro-categoria.
* **Feature Importance:** Analisi delle feature di rete più rilevanti (es. `Duration`, `rst_count`, `Header_Length`).

## 👥 Autori

Progetto sviluppato nell'ambito del corso di *Machine Learning per la Sicurezza Informatica*.

* **Sviluppatori:** [I tuoi nomi qui]
* **Riferimenti:** [CICIoT2023 Paper](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

---

*Ultimo aggiornamento: Gennaio 2026*


-   -   -   -
# 📊 Architettura Multi-Dataset per NIDS

Sistema di gestione dati professionale con 3 dataset separati per scopi distinti.

## 🎯 Panoramica Architettura

```
data/processed/
├── original/              # ✅ Dataset baseline (distribuzione naturale)
│   ├── train_processed.pkl
│   ├── test_processed.pkl
│   ├── val_processed.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── mapping_info.json
│
├── smote/                 # ✅ Dataset bilanciato con SMOTE
│   ├── train_smote.pkl         # ← Train con synthetic samples
│   ├── test_processed.pkl      # ← Test ORIGINALE (NO SMOTE)
│   ├── val_processed.pkl       # ← Val ORIGINALE (NO SMOTE)
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── mapping_info.json
│   └── smote_info.json         # ← Info su SMOTE applicato
│
└── production/            # ✅ Traffico reale catturato
    ├── production_capture1_20250115_143022.pkl
    ├── production_capture1_20250115_143022_metadata.json
    ├── production_capture2_20250116_091534.pkl
    └── production_capture2_20250116_091534_metadata.json
```

---

## 📁 Dataset 1: Original (Baseline)

### 🎯 Scopo
Dataset di **riferimento** con distribuzione naturale (non modificata).

### ✅ Caratteristiche
- Train/Test/Val dal dataset CICIoT2023 originale
- Distribuzione sbilanciata (~97% attacchi, ~3% benign)
- **Nessuna modifica** ai dati (solo pulizia e normalizzazione)
- Baseline per confronti

### 📊 Uso
- **Baseline experiments**: modello di riferimento
- **Confronti**: valutare impatto di SMOTE e altri metodi
- **Riproducibilità**: dataset immutabile per esperimenti ripetibili

### 🛠️ Creazione
```bash
python src/data_processing.py \
    --train-path data/raw/CICIOT23/train/train.csv \
    --test-path data/raw/CICIOT23/test/test.csv \
    --val-path data/raw/CICIOT23/validation/validation.csv \
    --output-dir data/processed/original
```

### 📈 Distribuzione Tipica
```
Benign:      ~2.5%  (minoranza)
DDoS:       ~72.0%  (maggioranza)
DoS:        ~17.0%
Mirai:       ~6.0%
Recon:       ~0.7%
Web:         ~0.02% (minoranza estrema)
Spoofing:    ~1.0%
BruteForce:  ~0.03% (minoranza estrema)
```

---

## 📁 Dataset 2: SMOTE (Bilanciato)

### 🎯 Scopo
Dataset **bilanciato** con campioni sintetici per classi minoritarie.

### ✅ Caratteristiche
- **Train**: SMOTE applicato → classi bilanciate
- **Test**: ORIGINALE (NO SMOTE) → valutazione realistica
- **Val**: ORIGINALE (NO SMOTE) → valutazione realistica
- Artifacts identici al dataset original

### ⚠️ REGOLA CRITICA: SMOTE Solo su Train

```
❌ SBAGLIATO:
apply_smote(train + test + val)  # Data leakage!

✅ CORRETTO:
apply_smote(train)               # Solo train
keep_original(test)              # Test originale
keep_original(val)               # Val originale
```

**Perché?**
- SMOTE genera campioni **sintetici** (interpolazioni)
- Test/Val devono riflettere **distribuzione reale**
- Altrimenti: valutazione troppo ottimistica ❌

### 📊 Uso
- **Training robusto**: classi minoritarie meglio rappresentate
- **Recall migliorato**: modello impara meglio classi rare (Web, BruteForce)
- **Confronto A/B**: original vs SMOTE → quale performa meglio?

### 🛠️ Creazione
```bash
# Prima crea dataset original
python src/data_processing.py --output-dir data/processed/original

# Poi applica SMOTE
python src/apply_smote.py \
    --input-dir data/processed/original \
    --output-dir data/processed/smote \
    --sampling-strategy auto \
    --k-neighbors 5
```

### 📈 Distribuzione Post-SMOTE
```
BEFORE SMOTE:                AFTER SMOTE:
Benign:      250 (2.5%)  →  Benign:      7,238 (balanced)
DDoS:      7,238 (72%)   →  DDoS:        7,238 (unchanged)
DoS:       1,718 (17%)   →  DoS:         7,238 (balanced)
Mirai:       614 (6%)    →  Mirai:       7,238 (balanced)
Recon:        74 (0.7%)  →  Recon:       7,238 (balanced)
Web:           2 (0.02%) →  Web:         7,238 (balanced!)
Spoofing:    100 (1%)    →  Spoofing:    7,238 (balanced)
BruteForce:    3 (0.03%) →  BruteForce:  7,238 (balanced!)

Total: 9,999 → ~58,000 (5.8x increase)
```

### 🔍 Vantaggi SMOTE
1. ✅ **Recall migliorato** per classi rare (Web: 2→7238 samples!)
2. ✅ **Overfitting ridotto** su minoranze
3. ✅ **Modello più robusto** su tutte le classi
4. ✅ **Generalizzazione migliore**

### ⚠️ Potenziali Svantaggi
1. ❌ **Overfitting su sintetici**: se k_neighbors troppo piccolo
2. ❌ **Training più lento**: ~6x più campioni
3. ❌ **Memoria maggiore**: dataset più grande

---

## 📁 Dataset 3: Production (Traffico Reale)

### 🎯 Scopo
Traffico di rete **catturato in produzione** per monitoraggio continuo.

### ✅ Caratteristiche
- Traffico **reale** da deployment
- Timestamp-based naming (tracciabilità)
- Processato con **stessi artifacts** del training
- Può avere o non avere labels

### 📊 Uso Principale

#### 1. **Inference in Produzione**
```python
# Carica modello trained
model = joblib.load('models/best_model.pkl')

# Carica production traffic
df_prod = pd.read_pickle('data/processed/production/capture_today.pkl')
X_prod = df_prod[feature_cols].values

# Predict
predictions = model.predict(X_prod)
probabilities = model.predict_proba(X_prod)
```

#### 2. **Drift Detection**
```python
# Confronta distribuzione features
original_mean = X_train.mean(axis=0)
production_mean = X_prod.mean(axis=0)

drift_score = np.abs(original_mean - production_mean).mean()
if drift_score > threshold:
    print("⚠️ Data drift detected! Consider retraining.")
```

#### 3. **Model Retraining**
```python
# Accumula production data labeled (es. da SOC analyst)
production_labeled = pd.concat([
    pd.read_pickle('data/processed/production/week1_labeled.pkl'),
    pd.read_pickle('data/processed/production/week2_labeled.pkl'),
    # ...
])

# Merge con training originale
X_retrain = np.vstack([X_train, X_prod_labeled])
y_retrain = np.hstack([y_train, y_prod_labeled])

# Retrain
model.fit(X_retrain, y_retrain)
```

#### 4. **Performance Monitoring**
```python
# Se production data ha labels (es. post-incident)
df_prod = pd.read_pickle('data/processed/production/incident_labeled.pkl')

y_true = df_prod['y_macro_encoded'].values
y_pred = model.predict(X_prod)

# Metriche
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"Production Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Recall: {recall:.4f}")
```

### 🛠️ Creazione
```bash
# Processa traffico grezzo catturato
python src/process_production_traffic.py \
    --input captures/network_traffic_20250115.csv \
    --artifacts-dir data/processed/original \
    --output-dir data/processed/production

# Con labels (se disponibili, es. post-incident)
python src/process_production_traffic.py \
    --input incident_logs/attack_20250116.csv \
    --artifacts-dir data/processed/original \
    --output-dir data/processed/production \
    --label-col label
```

### 📋 Metadata
Ogni file production ha metadata associato:
```json
{
  "source_file": "network_traffic_20250115.csv",
  "processed_timestamp": "2025-01-15T14:30:22",
  "num_records": 10543,
  "num_features": 46,
  "has_labels": false,
  "deployment_location": "datacenter-1",
  "capture_duration_minutes": 60
}
```

---

## 🔄 Workflow Completo

### 1️⃣ Setup Iniziale (Una Volta)

```bash
# Step 1: Preprocessing baseline
python src/data_processing.py \
    --train-path data/raw/CICIOT23/train/train.csv \
    --test-path data/raw/CICIOT23/test/test.csv \
    --val-path data/raw/CICIOT23/validation/validation.csv \
    --output-dir data/processed/original

# Step 2: Crea versione SMOTE
python src/apply_smote.py \
    --input-dir data/processed/original \
    --output-dir data/processed/smote
```

### 2️⃣ Training & Evaluation

```bash
# Train su original (baseline)
python src/train_random_forest_multiclass.py \
    --data-dir data/processed/original \
    --output-dir docs/original \
    --model-path models/rf_original.pkl

# Train su SMOTE (experimental)
python src/train_random_forest_multiclass.py \
    --data-dir data/processed/smote \
    --output-dir docs/smote \
    --model-path models/rf_smote.pkl

# Confronta risultati
python src/compare_models.py \
    --model1 models/rf_original.pkl \
    --model2 models/rf_smote.pkl \
    --test-data data/processed/original/test_processed.pkl
```

### 3️⃣ Production Deployment

```bash
# Cattura traffico (esempio con tcpdump)
tcpdump -i eth0 -w capture.pcap -G 3600 -W 1

# Converti PCAP → CSV (con CICFlowMeter o similar)
cicflowmeter -f capture.pcap -c output.csv

# Processa per inference
python src/process_production_traffic.py \
    --input output.csv \
    --artifacts-dir data/processed/original \
    --output-dir data/processed/production

# Run inference
python src/run_inference.py \
    --model models/rf_smote.pkl \
    --data data/processed/production/production_output_*.pkl
```

### 4️⃣ Continuous Monitoring

```bash
# Setup cron job per monitoraggio orario
# /etc/cron.d/nids-monitor
0 * * * * /path/to/monitor_drift.py

# Controllo settimanale performance
python src/evaluate_production.py \
    --model models/rf_smote.pkl \
    --production-dir data/processed/production \
    --week-range 2025-01-08:2025-01-14
```

---

## 📊 Confronto Strategico

| Aspetto | Original | SMOTE | Production |
|---------|----------|-------|------------|
| **Distribuzione** | Naturale (sbilanciata) | Bilanciata (sintetica) | Variabile (reale) |
| **Train Set** | Originale | Sintetici aggiunti | N/A |
| **Test Set** | Originale | Originale | Variabile |
| **Val Set** | Originale | Originale | N/A |
| **Size** | Base | ~6x più grande | Variabile |
| **Uso** | Baseline | Training robusto | Inference live |
| **Labels** | Sempre | Sempre | Opzionale |
| **Modifiche** | Solo pulizia | SMOTE su train | Processing standard |

---

## 🎯 Decision Tree: Quale Dataset Usare?

```
Domanda: Quale dataset usare?
│
├─ Sto facendo esperimenti iniziali?
│  └─ ✅ USA: original (baseline)
│
├─ Voglio migliorare recall su classi rare?
│  └─ ✅ USA: smote (bilanciato)
│
├─ Devo fare inference su traffico live?
│  └─ ✅ USA: production (real-time)
│
├─ Voglio confrontare approcci?
│  └─ ✅ USA: original vs smote (A/B test)
│
└─ Devo monitorare performance in produzione?
   └─ ✅ USA: production (labeled post-incident)
```

---

## 💡 Best Practices

### ✅ DO:
1. **Sempre** valuta su original test set (anche se trained su SMOTE)
2. **Traccia** quale dataset usato per ogni esperimento
3. **Mantieni** artifacts sincronizzati tra dataset
4. **Monitora** drift su production data
5. **Documenta** decisioni su quale dataset usare

### ❌ DON'T:
1. **Mai** applicare SMOTE a test/val
2. **Mai** mischiare original e SMOTE in stesso training
3. **Mai** valutare su dati sintetici
4. **Mai** ignorare production drift
5. **Mai** sovrascrivere dataset original

---

## 📈 Expected Results

### Scenario 1: Training su Original
```
Test Metrics (Original Test Set):
  Accuracy:  0.94  ❌ (sotto target 0.95)
  Precision: 0.92  ✅
  Recall:    0.89  ❌ (sotto target 0.95)

Problem: Classi minoritarie (Web, BruteForce) mal classificate
```

### Scenario 2: Training su SMOTE
```
Test Metrics (Original Test Set):
  Accuracy:  0.97  ✅
  Precision: 0.95  ✅
  Recall:    0.96  ✅

Per-Class Recall:
  Benign:      0.98  ✅
  DDoS:        0.99  ✅
  DoS:         0.97  ✅
  Mirai:       0.96  ✅
  Recon:       0.93  ✅
  Web:         0.85  ⚠️  (migliorato da 0.50!)
  Spoofing:    0.94  ✅
  BruteForce:  0.79  ⚠️  (migliorato da 0.33!)

Result: SMOTE significantly improves minority classes!
```

---

## 🔧 Troubleshooting

### Q: SMOTE peggiora le performance?
**A**: Possibile se:
- k_neighbors troppo piccolo → overfitting
- Classe maggioritaria già domina → dilution
- **Soluzione**: Tuning k_neighbors, prova sampling_strategy='minority'

### Q: Production drift rilevato?
**A**: Azioni:
1. Investigare cause (nuovo tipo attacco? cambio rete?)
2. Colleziona production data labeled
3. Retrain con production data
4. Re-deploy modello aggiornato

### Q: Quale dataset per model finale?
**A**: **SMOTE** se:
- Recall su minority classes è priorità
- Performance su test SMOTE > original
**Original** se:
- Dataset già relativamente bilanciato
- Performance simili ma original più veloce

---

## 📚 Riferimenti

- **SMOTE Paper**: Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
- **Imbalanced-learn**: https://imbalanced-learn.org/
- **Best Practices**: https://developers.google.com/machine-learning/data-prep

---

**Conclusione**: Questa architettura multi-dataset permette **flessibilità**, **riproducibilità**, e **production readiness** simultaneamente. È una best practice professionale per sistemi ML in produzione.