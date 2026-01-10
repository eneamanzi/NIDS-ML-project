
```markdown
# NIDS ML Project (CICIoT2023)

Sistema di Intrusion Detection (NIDS) basato su Machine Learning per ambienti IoT, utilizzando il dataset CICIoT2023. Il progetto implementa una pipeline completa: preprocessing ottimizzato (Parquet), data augmentation avanzata (SMOTE, ADASYN, CTGAN), training di multipli modelli, sniffing real-time e dashboard di monitoraggio.

## 📋 Struttura del Progetto

```text
.
├── data/               # Contiene i dati raw e processed (generati dagli script)
├── docs/               # Grafici e metriche generati durante il training
├── models/             # Modelli .pkl salvati
├── logs/               # Log di training e sniffer
├── src/                # Script sorgente (Preprocessing, Augmentation, Training, Sniffer)
└── dashboard/          # Dashboard Streamlit

```

## 🛠️ Installazione

Assicurati di avere Python 3.10+ installato.

1. **Installa le dipendenze:**
```bash
pip install -r requirements.txt

```


2. **Dataset:**
Scarica i file CSV del dataset CICIoT2023 e posizionali in `data/raw/CICIOT23/` (o modifica i percorsi di default negli script).

---

## 🚀 Workflow

La pipeline è progettata per essere eseguita in sequenza. Esegui tutti i comandi dalla **root** del progetto.

### 1. Preprocessing

Converte i CSV raw in formato Parquet ottimizzato (float32), gestisce le etichette e pulisce i dati.

```bash
python src/preprocessing.py

```

* **Parametri opzionali principali:**
* `--train-path`: Path del CSV di training raw (Default: `data/raw/CICIOT23/train/train.csv`).
* `--output-dir`: Cartella di output (Default: `data/processed/CICIOT23`).
* `--chunk-size`: Dimensione chunk per la lettura (Default: `500000`).



### 2. Data Augmentation (Opzionale)

Genera campioni sintetici per bilanciare le classi minoritarie. Puoi scegliere tra diverse strategie.

**ADASYN (Consigliato per densità adattiva):**

```bash
python src/adasyn.py

```

**Borderline-SMOTE (Focus sui bordi decisionali):**

```bash
python src/borderline_smote.py

```

**CTGAN (Deep Learning - Richiede risorse elevate):**

```bash
python src/ctgan_data_aug.py --subset-size 100000 --epochs 100

```

* **Nota sui parametri:** Tutti gli script di augmentation accettano `--target-sample` (default 800k) e `--min-minority` (default 50k) per controllare il bilanciamento.

### 3. Training dei Modelli

Puoi allenare i modelli singolarmente o utilizzare l'orchestratore.

**Training Singolo (Esempi):**
Ogni script supporta l'argomento `--dataset-type` che indica su quali dati allenare (`original`, `smote`, `borderline`, `adasyn`, `ctgan`).

* **Random Forest:**
```bash
python src/train_random_forest.py --dataset-type original
# Oppure su dati aumentati
python src/train_random_forest.py --dataset-type adasyn --n-estimators 200

```


* **XGBoost:**
```bash
python src/train_xgboost.py --dataset-type borderline

```


* **Decision Tree:**
```bash
python src/train_decision_tree.py --dataset-type original

```


* **MLP (Neural Network):**
```bash
python src/train_mlp.py --dataset-type original --hidden-layers 128,64

```



**Training Automatico (Orchestrator):**
Allena più modelli in sequenza.

```bash
# Allena tutto su dataset original e smote
python train_all_models.py

# Allena solo Random Forest su dataset original
python train_all_models.py --algorithms RandomForest --datasets original

```

### 4. Sniffer Real-Time

Analizza il traffico di rete in tempo reale utilizzando uno dei modelli addestrati.
*Richiede privilegi di root per lo sniffing live.*

```bash
# Modalità Live (sostituisci eth0 con la tua interfaccia)
sudo python src/sniffer.py --interface eth0 --model models/RandomForest/rf_model_original.pkl

# Modalità Offline (File PCAP)
python src/sniffer.py --pcap-file capture.pcap --model models/RandomForest/rf_model_original.pkl

```

* **Opzioni Firewall:**
* `--dry-run` (Default): Logga solo le minacce, non blocca nulla.
* `--firewall`: **Attiva il blocco IP** tramite `iptables` o `ufw` se rileva attacchi con alta confidenza.
* `--block-threshold`: Soglia di confidenza per il blocco (es. `0.9`).



### 5. Dashboard

Visualizza statistiche in tempo reale basate sull'output dello sniffer.

```bash
streamlit run dashboard/dashboard.py

```

La dashboard si aggiornerà automaticamente leggendo i dati dalla cartella `data/sniffer_output`.

---

## 📊 Output e Risultati

* **Grafici:** Matrici di confusione, curve di loss e feature importance vengono salvate in `docs/<Algoritmo>/<Dataset>/`.
* **Modelli:** I file `.pkl` addestrati sono salvati in `models/<Algoritmo>/`.
* **Log:** Controlla la cartella `logs/` per dettagli su training e sniffing.

```

```
