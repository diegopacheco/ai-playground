#!/bin/bash
set -e
python3 src/build_site.py
python3 -m http.server 8077 --directory site
