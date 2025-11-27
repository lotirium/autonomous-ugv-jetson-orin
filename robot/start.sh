#!/bin/bash
# Start Rovy Raspberry Pi Client
# Connects to PC server via Tailscale

cd "$(dirname "$0")"

echo "================================"
echo "  ROVY RASPBERRY PI CLIENT"
echo "================================"
echo "Server: ws://100.121.110.125:8765"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Check dependencies
python3 -c "import websockets" 2>/dev/null || {
    echo "Installing websockets..."
    pip3 install websockets
}

# Start client
python3 main.py

