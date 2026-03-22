#!/bin/bash

brew install espeak
rm -rf .venv
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
