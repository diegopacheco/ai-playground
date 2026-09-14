#!/usr/bin/env bash
set -euo pipefail
curl -s "https://api.open-meteo.com/v1/forecast?latitude=$1&longitude=$2&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&forecast_days=3"
