ATTACK_MAPPING = {
    # DDoS (7 varianti)
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    
    # DoS (4 varianti)
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    
    # Mirai (3 varianti)
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    
    # Spoofing (2 varianti)
    'MITM-ArpSpoofing': 'Spoofing',
    'DNS_Spoofing': 'Spoofing',
    
    # Recon (4 varianti)
    'Recon-HostDiscovery': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-PingSweep': 'Recon',
    'VulnerabilityScan': 'Recon',
    
    # Web (5 varianti)
    'SqlInjection': 'Web',
    'XSS': 'Web',
    'CommandInjection': 'Web',
    'Uploading_Attack': 'Web',
    'BrowserHijacking': 'Web',
    
    # BruteForce
    'DictionaryBruteForce': 'BruteForce',
    
    # Backdoor (categoria aggiuntiva)
    'Backdoor_Malware': 'Backdoor',
    
    # Benign
    'BenignTraffic': 'Benign'
}