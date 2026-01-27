import os
import paramiko
from scp import SCPClient
import subprocess
import time
from dotenv import load_dotenv

# 1. Load Secrets from .env file
load_dotenv()

# 2. Fetch Variables (Securely)
LINUX_HOST = os.getenv("LINUX_HOST")
LINUX_USER = os.getenv("LINUX_USER")
LINUX_PASSWORD = os.getenv("LINUX_PASSWORD")
REMOTE_DIR = os.getenv("REMOTE_DIR")

# Validation: Stop immediately if secrets are missing
if not all([LINUX_HOST, LINUX_USER, LINUX_PASSWORD, REMOTE_DIR]):
    print("   Please create a .env file with LINUX_HOST, LINUX_USER, LINUX_PASSWORD, REMOTE_DIR")
    exit(1)

def run_local_command(command):
    """Runs a command on the local Windows machine"""
    print(f"🔹 [LOCAL] Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        raise Exception("Local command failed")
    print("✅ Success")

def create_ssh_client():
    """Establishes SSH connection to Linux"""
    print(f"🔌 Connecting to {LINUX_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(LINUX_HOST, username=LINUX_USER, password=LINUX_PASSWORD)
    print(f"sucessfully connected to {LINUX_HOST}")
    return ssh

def execute_remote_command(ssh, command):
    """Runs a command on the remote Linux machine"""
    print(f"🔹 [REMOTE] Executing: {command}")
    stdin, stdout, stderr = ssh.exec_command(command)
   
    # Stream output
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line.strip())
       
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        print(f"❌ Remote Error: {stderr.read().decode()}")
        raise Exception("Remote command failed")
    print("✅ Remote Success")

def deploy():
    print("==========================================")
    print("🚀 STARTING SECURE DEPLOYMENT PIPELINE")
    print("==========================================")

    # 1. STEP 1: EXPORT MODEL
    print("\n--- Step 1: Exporting Model Artifacts ---")
    run_local_command("python src/export_model.py")

    # 2. STEP 2: CONNECT
    ssh = create_ssh_client()
    print(ssh)

    # 3. STEP 3: TRANSFER
    print("\n--- Step 3: Uploading Files to Linux ---")
    ssh.exec_command(f"mkdir -p {REMOTE_DIR}")
   
    with SCPClient(ssh.get_transport()) as scp:
        # Note: We DO NOT upload the .env file.
        # Production servers should have their own secrets managed separately.
        print("Uploading config, requirements, Dockerfile...")
        scp.put("config.yaml", remote_path=REMOTE_DIR)
        scp.put("requirements.txt", remote_path=REMOTE_DIR)
        scp.put("Dockerfile", remote_path=REMOTE_DIR)
       
        print("Uploading Source Code...")
        scp.put("src", recursive=True, remote_path=REMOTE_DIR)
       
        print("Uploading Model Artifacts...")
        scp.put("docker_model", recursive=True, remote_path=REMOTE_DIR)
       
    print("✅ Upload Complete.")

    # 4. STEP 4: DOCKER DEPLOY
    print("\n--- Step 4: Building & Deploying Docker Container ---")
   
    # Build
    build_cmd = f"cd {REMOTE_DIR} && echo '{LINUX_PASSWORD}' | sudo -S docker build -t telco-churn-api ."
    execute_remote_command(ssh, build_cmd)

    # Restart Container
    print("Restarting Container...")
    restart_cmd = (
        f"echo '{LINUX_PASSWORD}' | sudo -S docker stop my_churn_model || true && "
        f"echo '{LINUX_PASSWORD}' | sudo -S docker rm my_churn_model || true && "
        f"echo '{LINUX_PASSWORD}' | sudo -S docker run -d -p 8100:8100 --restart always --name my_churn_model telco-churn-api"
    )
    execute_remote_command(ssh, restart_cmd)

    ssh.close()
    print("\n==========================================")
    print(f"✅ DEPLOYMENT SUCCESSFUL!")
    print(f"🌍 API is live at: http://{LINUX_HOST}:8100/docs")
    print("==========================================")

if __name__ == "__main__":
    try:
        deploy()
    except Exception as e:
        print(f"\n PIPELINE FAILED: {e}")  