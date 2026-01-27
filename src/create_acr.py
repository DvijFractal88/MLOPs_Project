import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
LINUX_HOST = os.getenv("LINUX_HOST")
LINUX_USER = os.getenv("LINUX_USER")
LINUX_PASSWORD = os.getenv("LINUX_PASSWORD")

# Azure Credentials
SUB_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
RG_NAME = os.getenv("AZURE_RG_NAME")
LOCATION = os.getenv("AZURE_LOCATION")
ACR_NAME = os.getenv("AZURE_ACR_NAME")

if not all([LINUX_HOST, CLIENT_ID, RG_NAME, ACR_NAME]):
    print("❌ Error: Missing configuration in .env file.")
    exit(1)

def create_ssh_client():
    print(f"Connecting to Linux Server ({LINUX_HOST})...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(LINUX_HOST, username=LINUX_USER, password=LINUX_PASSWORD)

    return ssh

def execute_remote_command(ssh, command, description):
    print(f"\n🔹 {description}...")
    # 'set -e' stops the script if a command fails
    full_cmd = f"set -e; {command}"
    stdin, stdout, stderr = ssh.exec_command(full_cmd)
   
    # Stream the output
    while True:
        line = stdout.readline()
        if not line: break
        print(line.strip())
       
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        print(f"❌ Failed: {stderr.read().decode()}")
        raise Exception(f"Remote command failed: {description}")
    print("✅ Success")

def create_acr_remotely():
    print("==========================================")
    print("☁️  CREATING ACR (VIA LINUX SERVER)")
    print("==========================================")
   
    ssh = create_ssh_client()

    # 1. Login to Azure (On Linux)
    login_cmd = f'az login --service-principal -u "{CLIENT_ID}" -p "{CLIENT_SECRET}" --tenant "{TENANT_ID}"'
    execute_remote_command(ssh, login_cmd, "Logging Linux Server into Azure")

    # 2. Set Subscription
    # sub_cmd = f'az account set --subscription "{SUB_ID}"'
    # execute_remote_command(ssh, sub_cmd, "Setting Active Subscription")

    # 3. Create Resource Group (if needed)
    # rg_cmd = f'az group create --name "{RG_NAME}" --location "{LOCATION}"'
    # execute_remote_command(ssh, rg_cmd, f"Ensuring Resource Group '{RG_NAME}' exists")

    # 4. Create ACR
    # print(f"\n🔹 Creating Registry '{ACR_NAME}'...")
    # acr_cmd = (
    #     f'az acr create --resource-group "{RG_NAME}" '
    #     f'--name "{ACR_NAME}" '
    #     f'--sku Basic '
    #     f'--admin-enabled true '
    #     f'--location "{LOCATION}"'
    # )
    # execute_remote_command(ssh, acr_cmd, "Creating Azure Container Registry")

    ssh.close()
    print("\n==========================================")
    print(f"✅ ACR CREATED: {ACR_NAME}.azurecr.io")
    print("==========================================")

if __name__ == "__main__":
    create_acr_remotely()