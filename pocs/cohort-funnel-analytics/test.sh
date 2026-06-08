#!/bin/bash
set -e
python3 src/generate_events.py
python3 src/funnel.py
