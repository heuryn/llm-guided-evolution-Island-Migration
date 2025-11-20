import os

ROOT_DIR = "/home/hice1/aganesan44/scratch/llm-guided-evolution-Island-Migration/"
HOSTNAME_DIR = os.path.join(ROOT_DIR, "hostname.log")

def get_llm_server_hostnames():
    hostnames = []
    hostname_file_path = HOSTNAME_DIR 
    with open(hostname_file_path, 'r') as f:
        for line in f:
            hostnames.append(line.strip())
    return hostnames

print(get_llm_server_hostnames())