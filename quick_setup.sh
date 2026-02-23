#!/bin/bash
# Quick Setup Script for PharmPapp MCP
# PharmPapp: Peptide Permeability Prediction across cell lines, structural types, and modifications
# Supports Caco-2, RRCK, and PAMPA permeability prediction for linear and cyclic peptides
# Source: https://github.com/ifyoungnet/PharmPapp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up PharmPapp MCP ==="

# Step 1: Create Python environment
echo "[1/4] Creating Python 3.10 environment..."
(command -v mamba >/dev/null 2>&1 && mamba create -p ./env python=3.10 -y) || \
(command -v conda >/dev/null 2>&1 && conda create -p ./env python=3.10 -y) || \
(echo "Warning: Neither mamba nor conda found, creating venv instead" && python3 -m venv ./env)

# Step 2: Install core dependencies
echo "[2/4] Installing core dependencies..."
(command -v mamba >/dev/null 2>&1 && mamba run -p ./env pip install pandas numpy scikit-learn matplotlib seaborn tqdm loguru click) || \
./env/bin/pip install pandas numpy scikit-learn matplotlib seaborn tqdm loguru click

# Step 3: Install RDKit
echo "[3/4] Installing RDKit..."
(command -v mamba >/dev/null 2>&1 && mamba install -p ./env -c conda-forge rdkit -y) || \
./env/bin/pip install rdkit

# Step 4: Install fastmcp
echo "[4/4] Installing fastmcp..."
./env/bin/pip install --force-reinstall --no-cache-dir fastmcp

echo ""
echo "=== PharmPapp MCP Setup Complete ==="
echo "Note: PharmPapp uses KNIME workflows - see repo README for KNIME setup"
echo "To run the MCP server: ./env/bin/python src/server.py"
