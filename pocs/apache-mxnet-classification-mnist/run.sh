#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
python3 src/main.py
deactivate