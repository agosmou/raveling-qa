#!/bin/bash

# Exit on any error
set -e

# Function for timestamp logging
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to run Docker deployment
run_docker() {
    log_message "Starting Docker deployment..."
    if [ -f "docker-compose.yml" ]; then
        docker compose up -d
        log_message "Docker containers started and running Streamlit app"
    else
        log_message "Error: docker-compose.yml not found"
        exit 1
    fi
}

# Function to run local deployment
run_local() {
    log_message "Starting local deployment..."
    
    log_message "Changing directory to 'app/'"
    cd app/
    
    if [ -d ".venv" ]; then
        log_message "Cleaning up existing virtual environment"
        rm -rf .venv
    fi
    
    log_message "Creating virtual environment"
    python3 -m venv .venv
    
    log_message "Activating virtual environment"
    source .venv/bin/activate
    
    log_message "Installing requirements"
    pip install -r requirements.txt
    
    log_message "Starting Streamlit app locally"
    nohup streamlit run app.py > streamlit.log 2>&1 &
}

main() {
    # assumes firewall port 80 is accessible

    # can use local install if python 3.10, otherwise use docker
    # run_local

    run_docker

    log_message "Success! App is running locally"
}

main
