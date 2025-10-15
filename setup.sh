#!/bin/bash

set -e

ROOT_DIR=$(pwd)

echo "Setting up project structure under: $ROOT_DIR"

mkdir -p "$ROOT_DIR/data" \
         "$ROOT_DIR/output" 

echo "Warning, your laptop is ready to explode in"
sleep 1
echo "3"
sleep 1
echo "2"
sleep 1
echo "STAY AWAY FROM IT!!!"
sleep 1
echo "1"
sleep 1
echo "BOOM!!!"
sleep 1
echo " ... "
sleep 1
echo "HAHAHA, project structure created successfully."

Real_file="FitMRI_fitbit_intraday_steps_trainingData.csv"
DATA_DIR="$ROOT_DIR/data"

if [ -f "$Real_file" ]; then
    echo "Found $Real_file in current directory."
    echo "Copying $Real_file to data directory."
    cp "$Real_file" "$DATA_DIR"
    echo "Ok, I got your file."

else
    echo "Error: $Real_file not found in current directory."
    echo " Wait, why you don't have this file?"
    exit 1
fi

echo "Setup complete, and I wish your laptop survives."
echo "Use this to Run: chmod +x setup.sh"
echo "Then use this to Run: ./setup.sh"