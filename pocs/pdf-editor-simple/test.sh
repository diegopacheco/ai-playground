#!/bin/bash
set -e

./.venv/bin/python -m unittest discover -s . -p "test_*.py" -v
