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