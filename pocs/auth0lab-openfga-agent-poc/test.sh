#!/bin/bash
set -e

PYTHONPYCACHEPREFIX=.pycache python3 -m py_compile app.py tests.py
PYTHONDONTWRITEBYTECODE=1 python3 tests.py
PYTHONDONTWRITEBYTECODE=1 python3 app.py --user user:beth --query "token roadmap approval"
