#!/bin/bash

# Quick Start Script for Network Intrusion Detection System

echo "======================================================"
echo "🛡️  Network Intrusion Detection System"
echo "======================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if files exist
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found!"
    echo "Please ensure all files are in the current directory"
    exit 1
fi

# Set JAVA_HOME if not set
if [ -z "$JAVA_HOME" ]; then
    echo "⚙️  Setting JAVA_HOME..."
    export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
fi

echo ""
echo "🚀 Starting Network IDS application..."
echo ""
echo "The application will open in your browser at:"
echo "http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================================"
echo ""

# Run Streamlit
streamlit run app.py
