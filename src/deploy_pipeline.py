import os
import subprocess
import sys
import time
from dotenv import load_dotenv
from utils import Core_Operations

load_dotenv()
config = Core_Operations().load_config()

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

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def run_command(command, error_msg="Command failed"):
    """Runs a shell command and stops script if it fails."""
    print(f"🔹 Executing: {command}...")
    try:
        # shell=True is safe here since we control the inputs
        subprocess.check_call(command, shell=True, executable='/bin/bash')
        print("✅ Success.\n")
    except subprocess.CalledProcessError:
        print(f"❌ ERROR: {error_msg}")
        sys.exit(1)

def get_acr_password(acr_name, resource_group):
    """Fetches the ACR admin password for ACI deployment."""
    cmd = f"az acr credential show --name {acr_name} --resource-group {resource_group} --query 'passwords[0].value' -o tsv"
    try:
        password = subprocess.check_output(cmd, shell=True, executable='/bin/bash').decode('utf-8').strip()
        return password
    except Exception as e:
        print(f"❌ Failed to get ACR password: {e}")
        sys.exit(1)

# ==========================================
# 🚀 MAIN PIPELINE
# ==========================================
def main():
    print("🚀 STARTING AUTOMATED DEPLOYMENT PIPELINE (LINUX MODE)\n")

    # 1. Login to Azure (Service Principal)
    print("--- Step 1: Logging into Azure ---")
    login_cmd = (
        f"az login --service-principal "
        f"-u {CLIENT_ID} "
        f"-p '{CLIENT_SECRET}' "  # Quotes handle special chars
        f"--tenant {TENANT_ID}"
    )
    run_command(login_cmd, "Azure Login Failed. Check your SPN credentials.")

    # 2. Create Registry (ACR)
    print("--- Step 2: Ensuring ACR Exists ---")
    acr_cmd = (
        f"az acr create --resource-group {RG_NAME} "
        f"--name {config["deploy"]['ACR_NAME']} --sku Basic --admin-enabled true"
    )
    # We allow this to 'fail' if it already exists, so we wrap in try/except or just ignore stderr
    try:
        run_command(acr_cmd)
    except:
        print("⚠️ ACR might already exist. Continuing...")

    # 3. Login to ACR (Docker)
    print("--- Step 3: Logging Docker into ACR ---")
    run_command(f"az acr login --name {config["deploy"]['ACR_NAME']}", "Failed to login to ACR.")

    # 4. Build, Tag & Push Docker Image
    print("--- Step 4: Building & Pushing Image ---")
    full_image_tag = f"{config["deploy"]['ACR_NAME']}.azurecr.io/{config["deploy"]['IMAGE_NAME']}:{config["deploy"]['IMAGE_TAG']}"
   
    # Build (Offline mode baked in)
    run_command(f"docker build -t {config["deploy"]['IMAGE_NAME']} .", "Docker Build Failed.")
   
    # Tag
    run_command(f"docker tag {config["deploy"]['IMAGE_NAME']} {full_image_tag}", "Docker Tag Failed.")
   
    # Push
    run_command(f"docker push {full_image_tag}", "Docker Push Failed.")

    # 5. Deploy to ACI
    print("--- Step 5: Deploying Container Instance (ACI) ---")
   
    # Get Credentials for ACI to pull image
    acr_password = get_acr_password(config["deploy"]['ACR_NAME'], config["deploy"]['RESOURCE_GROUP'])

    # Delete old container if exists (Clean slate)
    delete_cmd = f"az container delete --resource-group {config["deploy"]['RESOURCE_GROUP']} --name {config["deploy"]['ACI_NAME']} --yes"
    subprocess.call(delete_cmd, shell=True, executable='/bin/bash') # Ignore error if it doesn't exist

    # Create new container
    deploy_cmd = (
        f"az container create --resource-group {config["deploy"]['RESOURCE_GROUP']} "
        f"--name {config["deploy"]['ACI_NAME']} "
        f"--image {full_image_tag} "
        f"--cpu 1 --memory 1.5 "
        f"--registry-login-server {config["deploy"]['ACR_NAME']}.azurecr.io "
        f"--registry-username {config["deploy"]['ACR_NAME']} "
        f"--registry-password {acr_password} "
        f"--dns-name-label {config["deploy"]['DNS_LABEL']} "
        f"--ports {config["deploy"]['APP_PORT']} "
        f"--location {config["deploy"]['LOCATION']} "
        f"--environment-variables IS_DOCKER=True"
    )
    run_command(deploy_cmd, "ACI Deployment Failed.")

    print("\n✅ DEPLOYMENT COMPLETE!")
    print(f"🌍 Your API is live at: http://{config["deploy"]['DNS_LABEL']}.{config["deploy"]['LOCATION']}.azurecontainer.io:{config["deploy"]['APP_PORT']}/docs")

if __name__ == "__main__":
    main()