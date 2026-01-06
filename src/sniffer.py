"""
NIDS Real-Time Network Sniffer con Integrazione Firewall.

Features:
- Live traffic sniffing O analisi file PCAP
- Predizione real-time con modello trained
- Integrazione firewall (iptables/ufw) con dry-run mode
- Logging completo e dettagliato
- Export CSV formato CICIoT2023

Usage:
    # Live sniffing (richiede sudo)
    sudo python src/sniffer.py --interface eth0 --model models/RandomForest/rf_model_smote.pkl
    
    # Analizza PCAP
    python src/sniffer.py --pcap-file capture.pcap --model models/RandomForest/rf_model_smote.pkl
    
    # Dry-run (no firewall modification)
    sudo python src/sniffer.py --interface eth0 --model models/best_model.pkl --dry-run
    
    # Con firewall attivo
    sudo python src/sniffer.py --interface eth0 --model models/best_model.pkl --firewall
"""

import sys
import os
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import subprocess
import signal

# Scapy imports
try:
    from scapy.all import sniff, wrpcap, rdpcap, IP, TCP, UDP, ICMP, Ether
    from scapy.layers.inet import TCP as TCP_LAYER, UDP as UDP_LAYER
except ImportError:
    print("❌ Scapy not installed. Install: pip install scapy")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs" / "sniffer"
OUTPUT_DIR = BASE_DIR / "data" / "sniffer_output"

# Create directories
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature names (CICIoT2023 format)
CICIOT_FEATURES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv',
    'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT',
    'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level='INFO', log_file=None):
    """Setup logging con file e console."""
    
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOGS_DIR / f'sniffer_{timestamp}.log'
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)  # File gets everything
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)
    
    return logger, log_file


# =============================================================================
# FLOW AGGREGATOR
# =============================================================================

class FlowAggregator:
    """
    Aggrega pacchetti in flussi per feature extraction.
    """
    
    def __init__(self, timeout=60):
        """
        Args:
            timeout: Secondi dopo cui un flusso è considerato completo
        """
        self.flows = {}  # {flow_id: flow_data}
        self.timeout = timeout
        self.logger = logging.getLogger('FlowAggregator')
    
    def get_flow_id(self, packet):
        """Genera flow ID univoco da pacchetto."""
        if IP not in packet:
            return None
        
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        
        if TCP in packet:
            proto = 'TCP'
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            proto = 'UDP'
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            proto = 'OTHER'
            src_port = 0
            dst_port = 0
        
        # Ordina per avere stesso ID per flusso bidirezionale
        endpoints = sorted([
            (src_ip, src_port),
            (dst_ip, dst_port)
        ])
        
        flow_id = f"{proto}_{endpoints[0][0]}:{endpoints[0][1]}_" \
                  f"{endpoints[1][0]}:{endpoints[1][1]}"
        
        return flow_id
    
    def add_packet(self, packet, timestamp):
        """Aggiunge pacchetto a flusso."""
        flow_id = self.get_flow_id(packet)
        
        if flow_id is None:
            return None
        
        if flow_id not in self.flows:
            self.flows[flow_id] = {
                'flow_id': flow_id,
                'packets': [],
                'start_time': timestamp,
                'last_time': timestamp,
                'src_ip': packet[IP].src if IP in packet else None,
                'dst_ip': packet[IP].dst if IP in packet else None,
                'protocol': self._get_protocol(packet),
                'flags': defaultdict(int),
                'packet_sizes': [],
                'iats': []  # Inter-arrival times
            }
        
        flow = self.flows[flow_id]
        flow['packets'].append(packet)
        flow['last_time'] = timestamp
        flow['packet_sizes'].append(len(packet))
        
        # Update IAT
        if len(flow['packets']) > 1:
            iat = timestamp - flow['last_time']
            flow['iats'].append(iat)
        
        # Update flags
        if TCP in packet:
            tcp = packet[TCP]
            if tcp.flags.F:
                flow['flags']['FIN'] += 1
            if tcp.flags.S:
                flow['flags']['SYN'] += 1
            if tcp.flags.R:
                flow['flags']['RST'] += 1
            if tcp.flags.P:
                flow['flags']['PSH'] += 1
            if tcp.flags.A:
                flow['flags']['ACK'] += 1
            if tcp.flags.U:
                flow['flags']['URG'] += 1
            if tcp.flags.E:
                flow['flags']['ECE'] += 1
            if tcp.flags.C:
                flow['flags']['CWR'] += 1
        
        return flow_id
    
    def _get_protocol(self, packet):
        """Determina protocollo."""
        if TCP in packet:
            return 'TCP'
        elif UDP in packet:
            return 'UDP'
        elif ICMP in packet:
            return 'ICMP'
        else:
            return 'OTHER'
    
    def get_completed_flows(self, current_time):
        """Restituisce flussi completati (timeout raggiunto)."""
        completed = []
        to_remove = []
        
        for flow_id, flow in self.flows.items():
            if current_time - flow['last_time'] >= self.timeout:
                completed.append(flow)
                to_remove.append(flow_id)
        
        # Remove
        for flow_id in to_remove:
            del self.flows[flow_id]
        
        if to_remove:
            self.logger.debug(f"Completed {len(completed)} flows")
        
        return completed
    
    def force_complete_all(self):
        """Forza completamento di tutti i flussi."""
        completed = list(self.flows.values())
        self.flows.clear()
        self.logger.info(f"Forced completion of {len(completed)} flows")
        return completed


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    """
    Estrae features da flussi nel formato CICIoT2023.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('FeatureExtractor')
    
    def extract_features(self, flow):
        """
        Estrae features da singolo flusso.
        
        Returns:
            dict con 46 features
        """
        packets = flow['packets']
        
        if len(packets) == 0:
            self.logger.warning(f"Empty flow: {flow['flow_id']}")
            return None
        
        # Duration
        duration = flow['last_time'] - flow['start_time']
        
        # Packet sizes
        sizes = flow['packet_sizes']
        total_size = sum(sizes)
        
        # IATs
        iats = flow['iats'] if flow['iats'] else [0]
        
        # Flags
        flags = flow['flags']
        
        # Protocol
        protocol = flow['protocol']
        
        features = {
            # Basic
            'flow_duration': duration,
            'Header_Length': np.mean([40 for _ in packets]),  # Approssimazione
            'Protocol Type': 6 if protocol == 'TCP' else 17 if protocol == 'UDP' else 1,
            'Duration': duration,
            
            # Rates
            'Rate': len(packets) / max(duration, 0.001),
            'Srate': len(packets) / 2 / max(duration, 0.001),  # Approssimazione
            'Drate': len(packets) / 2 / max(duration, 0.001),
            
            # Flags
            'fin_flag_number': flags.get('FIN', 0),
            'syn_flag_number': flags.get('SYN', 0),
            'rst_flag_number': flags.get('RST', 0),
            'psh_flag_number': flags.get('PSH', 0),
            'ack_flag_number': flags.get('ACK', 0),
            'ece_flag_number': flags.get('ECE', 0),
            'cwr_flag_number': flags.get('CWR', 0),
            
            # Counts
            'ack_count': flags.get('ACK', 0),
            'syn_count': flags.get('SYN', 0),
            'fin_count': flags.get('FIN', 0),
            'urg_count': flags.get('URG', 0),
            'rst_count': flags.get('RST', 0),
            
            # Protocols (binary)
            'HTTP': 1 if self._is_http(packets) else 0,
            'HTTPS': 1 if self._is_https(packets) else 0,
            'DNS': 1 if self._is_dns(packets) else 0,
            'Telnet': 0,  # Difficile da rilevare
            'SMTP': 0,
            'SSH': 1 if self._is_ssh(packets) else 0,
            'IRC': 0,
            'TCP': 1 if protocol == 'TCP' else 0,
            'UDP': 1 if protocol == 'UDP' else 0,
            'DHCP': 0,
            'ARP': 0,
            'ICMP': 1 if protocol == 'ICMP' else 0,
            'IPv': 4,  # IPv4
            'LLC': 0,
            
            # Stats
            'Tot sum': total_size,
            'Min': min(sizes) if sizes else 0,
            'Max': max(sizes) if sizes else 0,
            'AVG': np.mean(sizes) if sizes else 0,
            'Std': np.std(sizes) if sizes else 0,
            'Tot size': total_size,
            'IAT': np.mean(iats) if iats else 0,
            'Number': len(packets),
            
            # Advanced stats (approssimazioni)
            'Magnitue': np.sqrt(total_size),
            'Radius': np.mean(sizes) / 2 if sizes else 0,
            'Covariance': np.var(sizes) if len(sizes) > 1 else 0,
            'Variance': np.var(sizes) if len(sizes) > 1 else 0,
            'Weight': total_size / len(packets) if packets else 0
        }
        
        return features
    
    def _is_http(self, packets):
        """Rileva HTTP."""
        for pkt in packets:
            if TCP in pkt and (pkt[TCP].sport == 80 or pkt[TCP].dport == 80):
                return True
        return False
    
    def _is_https(self, packets):
        """Rileva HTTPS."""
        for pkt in packets:
            if TCP in pkt and (pkt[TCP].sport == 443 or pkt[TCP].dport == 443):
                return True
        return False
    
    def _is_dns(self, packets):
        """Rileva DNS."""
        for pkt in packets:
            if UDP in pkt and (pkt[UDP].sport == 53 or pkt[UDP].dport == 53):
                return True
        return False
    
    def _is_ssh(self, packets):
        """Rileva SSH."""
        for pkt in packets:
            if TCP in pkt and (pkt[TCP].sport == 22 or pkt[TCP].dport == 22):
                return True
        return False


# =============================================================================
# FIREWALL MANAGER
# =============================================================================

class FirewallManager:
    """
    Gestisce integrazione con firewall (iptables/ufw).
    """
    
    def __init__(self, dry_run=True):
        """
        Args:
            dry_run: Se True, solo logga azioni senza modificare firewall
        """
        self.dry_run = dry_run
        self.logger = logging.getLogger('FirewallManager')
        self.blocked_ips = set()
        
        # Detect firewall type
        self.firewall_type = self._detect_firewall()
        
        if self.dry_run:
            self.logger.warning("DRY-RUN MODE: Firewall actions will be logged but NOT executed")
        else:
            self.logger.warning("LIVE MODE: Firewall will be modified!")
    
    def _detect_firewall(self):
        """Rileva tipo firewall disponibile."""
        # Check ufw
        try:
            result = subprocess.run(['which', 'ufw'], 
                                   capture_output=True, timeout=5)
            if result.returncode == 0:
                self.logger.info("Detected firewall: ufw")
                return 'ufw'
        except:
            pass
        
        # Check iptables
        try:
            result = subprocess.run(['which', 'iptables'], 
                                   capture_output=True, timeout=5)
            if result.returncode == 0:
                self.logger.info("Detected firewall: iptables")
                return 'iptables'
        except:
            pass
        
        self.logger.warning("No firewall detected (ufw/iptables)")
        return None
    
    def block_ip(self, ip_address, reason="Malicious traffic detected"):
        """
        Blocca IP address.
        
        Args:
            ip_address: IP da bloccare
            reason: Motivo del blocco
        """
        if ip_address in self.blocked_ips:
            self.logger.debug(f"IP already blocked: {ip_address}")
            return False
        
        action_msg = f"BLOCK IP: {ip_address} | Reason: {reason}"
        
        if self.dry_run:
            self.logger.warning(f"[DRY-RUN] {action_msg}")
            self.blocked_ips.add(ip_address)
            return True
        
        # Execute firewall command
        if self.firewall_type == 'ufw':
            cmd = ['ufw', 'deny', 'from', ip_address]
        elif self.firewall_type == 'iptables':
            cmd = ['iptables', '-A', 'INPUT', '-s', ip_address, '-j', 'DROP']
        else:
            self.logger.error(f"Cannot block {ip_address}: No firewall available")
            return False
        
        try:
            self.logger.warning(f"[LIVE] {action_msg}")
            self.logger.debug(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, timeout=10, text=True)
            
            if result.returncode == 0:
                self.blocked_ips.add(ip_address)
                self.logger.info(f"✅ Successfully blocked: {ip_address}")
                return True
            else:
                self.logger.error(f"❌ Failed to block {ip_address}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout blocking {ip_address}")
            return False
        except Exception as e:
            self.logger.error(f"Error blocking {ip_address}: {e}")
            return False
    
    def unblock_ip(self, ip_address):
        """Sblocca IP (per testing)."""
        if ip_address not in self.blocked_ips:
            return False
        
        if self.dry_run:
            self.logger.info(f"[DRY-RUN] UNBLOCK IP: {ip_address}")
            self.blocked_ips.remove(ip_address)
            return True
        
        if self.firewall_type == 'ufw':
            cmd = ['ufw', 'delete', 'deny', 'from', ip_address]
        elif self.firewall_type == 'iptables':
            cmd = ['iptables', '-D', 'INPUT', '-s', ip_address, '-j', 'DROP']
        else:
            return False
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
            self.blocked_ips.remove(ip_address)
            self.logger.info(f"✅ Unblocked: {ip_address}")
            return True
        except:
            return False
    
    def get_blocked_count(self):
        """Numero IP bloccati."""
        return len(self.blocked_ips)


# Continua nella parte 2...


# =============================================================================
# NIDS SNIFFER (Main Class)
# =============================================================================

class NIDSSniffer:
    """
    Network Intrusion Detection System Sniffer.
    """
    
    def __init__(self, model_path, scaler_path=None, encoder_path=None,
                 dry_run=True, block_threshold=0.8):
        """
        Args:
            model_path: Path al modello trained
            scaler_path: Path allo scaler (optional, auto-detect)
            encoder_path: Path al label encoder (optional, auto-detect)
            dry_run: Firewall dry-run mode
            block_threshold: Soglia probabilità per bloccare IP (0-1)
        """
        self.logger = logging.getLogger('NIDSSniffer')
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load artifacts
        if scaler_path is None:
            scaler_path = BASE_DIR / "data" / "processed" / "CICIOT23" / "scaler.pkl"
        if encoder_path is None:
            encoder_path = BASE_DIR / "data" / "processed" / "CICIOT23" / "label_encoder.pkl"
        
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        
        self.logger.info(f"Model loaded: {model_path}")
        self.logger.info(f"Classes: {self.label_encoder.classes_}")
        
        # Components
        self.flow_aggregator = FlowAggregator(timeout=60)
        self.feature_extractor = FeatureExtractor()
        self.firewall = FirewallManager(dry_run=dry_run)
        
        # Config
        self.block_threshold = block_threshold
        
        # Stats
        self.stats = {
            'packets_processed': 0,
            'flows_completed': 0,
            'attacks_detected': 0,
            'ips_blocked': 0,
            'benign_flows': 0
        }
        
        # Data storage
        self.predictions = []
        self.running = True
    
    def _load_model(self, model_path):
        """Carica modello ML."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        self.logger.info(f"✅ Model loaded: {model_path.name}")
        
        return model
    
    def process_packet(self, packet):
        """Processa singolo pacchetto."""
        try:
            self.stats['packets_processed'] += 1
            
            # Add to flow
            timestamp = packet.time if hasattr(packet, 'time') else datetime.now().timestamp()
            flow_id = self.flow_aggregator.add_packet(packet, timestamp)
            
            # Check completed flows
            completed_flows = self.flow_aggregator.get_completed_flows(timestamp)
            
            for flow in completed_flows:
                self._process_flow(flow)
            
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
    
    def _process_flow(self, flow):
        """Processa flusso completato."""
        try:
            self.stats['flows_completed'] += 1
            
            # Extract features
            features = self.feature_extractor.extract_features(flow)
            
            if features is None:
                return
            
            # Prepare for prediction
            X = self._prepare_features(features)
            
            # Predict
            y_pred = self.model.predict(X)[0]
            y_proba = self.model.predict_proba(X)[0]
            
            # Decode label
            predicted_class = self.label_encoder.inverse_transform([y_pred])[0]
            confidence = np.max(y_proba)
            
            # Decision
            is_attack = (predicted_class != 'Benign')
            
            if is_attack:
                self.stats['attacks_detected'] += 1
                
                # Log attack
                self.logger.warning(
                    f"🚨 ATTACK DETECTED | "
                    f"Flow: {flow['flow_id']} | "
                    f"Class: {predicted_class} | "
                    f"Confidence: {confidence:.4f} | "
                    f"SrcIP: {flow['src_ip']}"
                )
                
                # Block if confidence > threshold
                if confidence >= self.block_threshold:
                    if self.firewall.block_ip(flow['src_ip'], 
                                             reason=f"{predicted_class} ({confidence:.2f})"):
                        self.stats['ips_blocked'] += 1
            else:
                self.stats['benign_flows'] += 1
                self.logger.debug(
                    f"✅ BENIGN | Flow: {flow['flow_id']} | "
                    f"Confidence: {confidence:.4f}"
                )
            
            # Store prediction
            self.predictions.append({
                'timestamp': datetime.fromtimestamp(flow['start_time']).isoformat(),
                'flow_id': flow['flow_id'],
                'src_ip': flow['src_ip'],
                'dst_ip': flow['dst_ip'],
                'protocol': flow['protocol'],
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_attack': is_attack,
                'blocked': is_attack and confidence >= self.block_threshold,
                **features
            })
            
        except Exception as e:
            self.logger.error(f"Error processing flow: {e}")
    
    def _prepare_features(self, features):
        """Prepara features per predizione."""
        # Ordina secondo CICIOT_FEATURES
        feature_vector = []
        for feat_name in CICIOT_FEATURES:
            feature_vector.append(features.get(feat_name, 0))
        
        # Convert to array e scala
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def sniff_live(self, interface='eth0', packet_count=0):
        """
        Sniff live traffic.
        
        Args:
            interface: Network interface
            packet_count: 0 = infinite
        """
        self.logger.info(f"🔍 Starting live sniffing on {interface}")
        self.logger.info(f"   Packet count: {'infinite' if packet_count == 0 else packet_count}")
        self.logger.info(f"   Block threshold: {self.block_threshold}")
        
        def stop_sniffer(sig, frame):
            self.logger.info("Stopping sniffer...")
            self.running = False
        
        signal.signal(signal.SIGINT, stop_sniffer)
        
        try:
            sniff(
                iface=interface,
                prn=self.process_packet,
                store=False,
                count=packet_count,
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            self.logger.error(f"Sniffing error: {e}")
        finally:
            self._finalize()
    
    def analyze_pcap(self, pcap_file):
        """
        Analizza file PCAP.
        
        Args:
            pcap_file: Path al file PCAP
        """
        pcap_file = Path(pcap_file)
        
        if not pcap_file.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
        
        self.logger.info(f"📂 Analyzing PCAP: {pcap_file}")
        
        try:
            packets = rdpcap(str(pcap_file))
            self.logger.info(f"   Total packets: {len(packets)}")
            
            for i, packet in enumerate(packets):
                if i % 1000 == 0 and i > 0:
                    self.logger.info(f"   Processed {i}/{len(packets)} packets...")
                
                self.process_packet(packet)
            
            # Force complete remaining flows
            remaining = self.flow_aggregator.force_complete_all()
            for flow in remaining:
                self._process_flow(flow)
            
        except Exception as e:
            self.logger.error(f"PCAP analysis error: {e}")
        finally:
            self._finalize()
    
    def _finalize(self):
        """Finalizza sniffing e salva risultati."""
        self.logger.info("Finalizing...")
        
        # Complete remaining flows
        remaining = self.flow_aggregator.force_complete_all()
        for flow in remaining:
            self._process_flow(flow)
        
        # Save results
        self._save_results()
        
        # Print stats
        self._print_stats()
    
    def _save_results(self):
        """Salva risultati in CSV."""
        if not self.predictions:
            self.logger.warning("No predictions to save")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = OUTPUT_DIR / f'predictions_{timestamp}.csv'
        
        df = pd.DataFrame(self.predictions)
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"💾 Results saved: {csv_file}")
        self.logger.info(f"   Records: {len(df)}")
    
    def _print_stats(self):
        """Stampa statistiche finali."""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("FINAL STATISTICS".center(80))
        self.logger.info("="*80)
        self.logger.info(f"  Packets processed:  {self.stats['packets_processed']:>10,}")
        self.logger.info(f"  Flows completed:    {self.stats['flows_completed']:>10,}")
        self.logger.info(f"  Benign flows:       {self.stats['benign_flows']:>10,}")
        self.logger.info(f"  Attacks detected:   {self.stats['attacks_detected']:>10,}")
        self.logger.info(f"  IPs blocked:        {self.stats['ips_blocked']:>10,}")
        
        if self.stats['flows_completed'] > 0:
            attack_rate = self.stats['attacks_detected'] / self.stats['flows_completed'] * 100
            self.logger.info(f"  Attack rate:        {attack_rate:>10.2f}%")
        
        self.logger.info("="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NIDS Real-Time Network Sniffer with Firewall Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live sniffing (requires sudo)
  sudo python src/sniffer.py --interface eth0 --model models/RandomForest/rf_model_smote.pkl
  
  # Analyze PCAP file
  python src/sniffer.py --pcap-file capture.pcap --model models/best_model.pkl
  
  # Dry-run mode (no firewall modification)
  sudo python src/sniffer.py --interface eth0 --model models/best_model.pkl --dry-run
  
  # With firewall active
  sudo python src/sniffer.py --interface eth0 --model models/best_model.pkl --firewall --block-threshold 0.9
        """
    )
    
    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--interface', type=str,
                           help='Network interface for live sniffing (requires sudo)')
    mode_group.add_argument('--pcap-file', type=Path,
                           help='PCAP file to analyze')
    
    # Model
    parser.add_argument('--model', type=Path, required=True,
                       help='Path to trained model (.pkl)')
    parser.add_argument('--scaler', type=Path,
                       help='Path to scaler (auto-detect if not specified)')
    parser.add_argument('--encoder', type=Path,
                       help='Path to label encoder (auto-detect if not specified)')
    
    # Firewall
    firewall_group = parser.add_mutually_exclusive_group()
    firewall_group.add_argument('--firewall', action='store_true',
                               help='Enable firewall modification (DANGEROUS!)')
    firewall_group.add_argument('--dry-run', action='store_true', default=True,
                               help='Dry-run mode: log actions without modifying firewall (default)')
    
    parser.add_argument('--block-threshold', type=float, default=0.8,
                       help='Confidence threshold for blocking IPs (0-1, default: 0.8)')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Other
    parser.add_argument('--packet-count', type=int, default=0,
                       help='Number of packets to capture (0=infinite, default: 0)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger, log_file = setup_logging(args.log_level)
    
    logger.info("="*80)
    logger.info("NIDS REAL-TIME SNIFFER".center(80))
    logger.info("="*80)
    logger.info("")
    
    # Check sudo for live sniffing
    if args.interface and os.geteuid() != 0:
        logger.error("Live sniffing requires sudo privileges!")
        logger.error("Run: sudo python src/sniffer.py ...")
        sys.exit(1)
    
    # Create sniffer
    try:
        sniffer = NIDSSniffer(
            model_path=args.model,
            scaler_path=args.scaler,
            encoder_path=args.encoder,
            dry_run=not args.firewall,  # dry_run è l'opposto di firewall
            block_threshold=args.block_threshold
        )
    except Exception as e:
        logger.error(f"Failed to initialize sniffer: {e}")
        sys.exit(1)
    
    # Run
    try:
        if args.interface:
            sniffer.sniff_live(
                interface=args.interface,
                packet_count=args.packet_count
            )
        else:
            sniffer.analyze_pcap(args.pcap_file)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"\n📁 Log file: {log_file}")
    logger.info("Goodbye! 👋")


if __name__ == '__main__':
    main()