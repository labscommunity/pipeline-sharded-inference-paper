#!/usr/bin/env bash
# setup_env.sh — verify that this machine has the dependencies needed to run
# the paper reproduction. Run on every node (rainier alpha/charlie/beta and any
# Tiber instance you'll use). Prints a checklist; exits non-zero if anything
# critical is missing.

set -uo pipefail
fail=0

echo "=== Reproduction package: environment check ==="
echo ""

# Python
if ! command -v python >/dev/null 2>&1 && ! command -v python.exe >/dev/null 2>&1; then
  echo "[FAIL] python not found"
  fail=1
else
  PY=$(command -v python || command -v python.exe)
  ver=$("$PY" --version 2>&1)
  echo "[ok]   $PY ($ver)"
fi

# OpenVINO
ov_ver=$("$PY" -c "import openvino; print(openvino.__version__)" 2>/dev/null)
if [ -z "$ov_ver" ]; then
  echo "[FAIL] openvino not importable. Install: pip install openvino==2026.1.0"
  fail=1
else
  echo "[ok]   openvino $ov_ver"
  case "$ov_ver" in
    2026.1*|2026.2*) ;;
    *) echo "[warn] paper used 2026.1.0+; you have $ov_ver — exact-equivalence not guaranteed" ;;
  esac
fi

# transformers
hf_ver=$("$PY" -c "import transformers; print(transformers.__version__)" 2>/dev/null)
if [ -z "$hf_ver" ]; then
  echo "[FAIL] transformers not importable"
  fail=1
else
  echo "[ok]   transformers $hf_ver"
fi

# torch
torch_ver=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ -z "$torch_ver" ]; then
  echo "[FAIL] torch not importable"
  fail=1
else
  echo "[ok]   torch $torch_ver"
fi

# nncf (used during INT4 export)
nncf_ver=$("$PY" -c "import nncf; print(nncf.__version__)" 2>/dev/null)
if [ -z "$nncf_ver" ]; then
  echo "[warn] nncf not importable — only needed for fresh exports, not for running benches"
else
  echo "[ok]   nncf $nncf_ver"
fi

# numpy
np_ver=$("$PY" -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [ -z "$np_ver" ]; then
  echo "[FAIL] numpy not importable"
  fail=1
else
  echo "[ok]   numpy $np_ver"
fi

# OV devices (informational)
echo ""
echo "=== OpenVINO devices ==="
"$PY" -c "import openvino as ov; c=ov.Core(); print('available:', c.available_devices)" 2>/dev/null || echo "(could not query)"

echo ""
if [ $fail -eq 0 ]; then
  echo "[OK] all critical dependencies present"
  exit 0
else
  echo "[!] one or more critical dependencies missing — install them then re-run"
  exit 1
fi
