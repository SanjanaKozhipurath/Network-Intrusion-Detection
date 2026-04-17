#!/bin/bash

# Installation Script for Network Intrusion Detection System
# Ubuntu 22.04

echo "======================================================"
echo "Network Intrusion Detection System - Setup"
echo "======================================================"
echo ""

# Check if running on Ubuntu
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" != "ubuntu" ]; then
        echo "⚠️  Warning: This script is designed for Ubuntu 22.04"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "❌ Cannot detect OS"
    exit 1
fi

echo "📦 Step 1: Update system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo ""
echo "📦 Step 2: Install Python 3 and pip..."
sudo apt-get install -y python3 python3-pip python3-venv

echo ""
echo "📦 Step 3: Install Java (required for Spark)..."
sudo apt-get install -y openjdk-11-jdk

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc

echo ""
echo "📦 Step 4: Create Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ""
echo "📦 Step 5: Upgrade pip..."
pip install --upgrade pip

echo ""
echo "📦 Step 6: Install Python dependencies..."
pip install streamlit pandas numpy matplotlib seaborn scikit-learn pyspark synapse-ml

echo ""
echo "📦 Step 7: Set up project directory..."
mkdir -p network_ids
cd network_ids

echo ""
echo "✅ Installation complete!"
echo ""
echo "======================================================"
echo "To start the application:"
echo "======================================================"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Navigate to project directory:"
echo "   cd network_ids"
echo ""
echo "3. Run the application:"
echo "   streamlit run app.py"
echo ""
echo "4. Open your browser to:"
echo "   http://localhost:8501"
echo "======================================================"
