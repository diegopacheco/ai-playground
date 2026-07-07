#!/bin/bash
set -e

python3 app.py --user user:beth --query "token roadmap approval architecture"
echo ""
python3 app.py --user user:carl --query "roadmap architecture"
echo ""
python3 app.py --user user:dana --query "payroll"
