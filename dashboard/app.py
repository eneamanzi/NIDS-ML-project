"""
Dashboard Flask per monitoraggio NIDS.
"""

from flask import Flask, render_template, jsonify
import joblib
import json
import os
from datetime import datetime
import sys

# Aggiungi src/ a path per imports
sys.path.append('../src')

app = Flask(__name__)

# Path files
ANOMALY_LOG = '../logs/anomalies.jsonl'
FIREWALL_LOG = '../logs/firewall.jsonl'
MODEL_DIR = '../models'


@app.route('/')
def index():
    """Homepage con monitor real-time."""
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    """
    Statistiche globali sistema.
    
    Returns:
        JSON con metriche
    """
    # Conta anomalie totali
    anomaly_count = 0
    if os.path.exists(ANOMALY_LOG):
        with open(ANOMALY_LOG, 'r') as f:
            anomaly_count = sum(1 for line in f)
    
    # Conta IP bloccati
    blocked_count = 0
    if os.path.exists(FIREWALL_LOG):
        with open(FIREWALL_LOG, 'r') as f:
            blocked_ips = set()
            for line in f:
                entry = json.loads(line)
                if entry.get('action') == 'BLOCKED':
                    blocked_ips.add(entry['ip'])
            blocked_count = len(blocked_ips)
    
    # Carica metriche modelli
    models_info = []
    model_files = ['DecisionTree.pkl', 'RandomForest.pkl', 'KNN.pkl', 'SVM.pkl']
    for model_file in model_files:
        path = f'{MODEL_DIR}/{model_file}'
        if os.path.exists(path):
            model_name = model_file.replace('.pkl', '')
            # Info base (accuracy hardcoded per ora, puoi caricare da file)
            models_info.append({
                'name': model_name,
                'accuracy': 0.97 if 'Random' in model_name else 0.94,  # Placeholder
                'status': 'active' if 'Random' in model_name else 'trained'
            })
    
    return jsonify({
        'total_anomalies': anomaly_count,
        'blocked_ips': blocked_count,
        'models': models_info,
        'uptime': '5h 23m',  # Placeholder, implementa tracking reale
        'status': 'running'
    })


@app.route('/api/recent_anomalies')
def get_recent_anomalies():
    """
    Ultime 50 anomalie rilevate.
    
    Returns:
        JSON array con anomalie
    """
    anomalies = []
    
    if os.path.exists(ANOMALY_LOG):
        with open(ANOMALY_LOG, 'r') as f:
            lines = f.readlines()
            # Ultime 50
            for line in lines[-50:]:
                anomalies.append(json.loads(line))
    
    # Ordina per timestamp decrescente
    anomalies.reverse()
    
    return jsonify(anomalies)


@app.route('/api/blocked_ips')
def get_blocked_ips():
    """
    Lista IP attualmente bloccati.
    
    Returns:
        JSON array con IP
    """
    blocked = []
    
    if os.path.exists(FIREWALL_LOG):
        with open(FIREWALL_LOG, 'r') as f:
            blocked_dict = {}
            for line in f:
                entry = json.loads(line)
                ip = entry['ip']
                if entry.get('action') == 'BLOCKED':
                    blocked_dict[ip] = entry
                # Se unblocked, rimuovi
                elif entry.get('action') == 'UNBLOCKED':
                    blocked_dict.pop(ip, None)
            
            blocked = list(blocked_dict.values())
    
    return jsonify(blocked)


@app.route('/api/traffic_chart')
def get_traffic_chart():
    """
    Dati per grafico traffico normale vs anomalo.
    
    Returns:
        JSON con time series
    """
    # Placeholder: genera dati fittizi per demo
    # In produzione, trackiamo contatori ogni minuto
    
    from datetime import timedelta
    now = datetime.now()
    
    data = []
    for i in range(10):
        timestamp = (now - timedelta(minutes=10-i)).strftime('%H:%M')
        data.append({
            'time': timestamp,
            'normal': 95 + (i % 3),
            'anomaly': 5 - (i % 3)
        })
    
    return jsonify(data)


# Run app
if __name__ == '__main__':
    # Crea logs se non esistono
    os.makedirs('../logs', exist_ok=True)
    
    print("="*60)
    print("NIDS DASHBOARD")
    print("="*60)
    print("Starting Flask server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)