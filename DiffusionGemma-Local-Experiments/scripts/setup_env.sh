#!/bin/bash
echo "Setting up DiffusionGemma Local Environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Environment ready. Activate it with: source venv/bin/activate"
