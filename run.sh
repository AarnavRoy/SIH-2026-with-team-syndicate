#!/bin/bash
echo "==================================================="
echo "  SignalScope - AI Media Forensics & Verification"
echo "  SIH 2026 [Internal Hackathon] Team Syndicate"
echo "==================================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    exit 1
fi

echo "[*] Checking dependencies..."
python3 -c "import fastapi, uvicorn, torch, torchvision, PIL, numpy" &> /dev/null
if [ $? -ne 0 ]; then
    echo "[*] Installing missing dependencies from requirements.txt..."
    pip3 install -r requirements.txt
else
    echo "[OK] All core dependencies verified."
fi

echo ""
echo "[*] Launching SignalScope on http://127.0.0.1:8000..."
# Attempt to open browser if desktop environment available
if which xdg-open > /dev/null; then
    (sleep 2 && xdg-open http://127.0.0.1:8000) &
elif which open > /dev/null; then
    (sleep 2 && open http://127.0.0.1:8000) &
fi

python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
