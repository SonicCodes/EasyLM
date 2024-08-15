#!/bin/bash

# Function to print usage
usage() {
    echo "Usage: $0 [--force_reinstall]"
    exit 1
}

# Parse command line arguments
FORCE_REINSTALL=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force_reinstall) FORCE_REINSTALL=true ;;
        *) usage ;;
    esac
    shift
done

# Set variables
TPU_NAME="LLaMA"
ZONE="us-central2-b"
REPO_URL="https://github.com/SonicCodes/EasyLM.git"

# Function to run command on all TPU VM workers
run_on_all_workers() {
    echo "Running command on all workers: $1"
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command "$1"
}

# Clone or update EasyLM repository
run_on_all_workers "
if [ ! -d \"EasyLM\" ]; then
    git clone $REPO_URL
else
    cd EasyLM
    git remote set-url origin $REPO_URL
    git pull
fi
"

# Run setup.sh on all workers
if [ "$FORCE_REINSTALL" = true ]; then
    run_on_all_workers "cd EasyLM && bash scripts/tpu_vm_setup.sh --force_reinstall"
else
    run_on_all_workers "cd EasyLM && bash scripts/tpu_vm_setup.sh"
fi

echo "All operations completed."
