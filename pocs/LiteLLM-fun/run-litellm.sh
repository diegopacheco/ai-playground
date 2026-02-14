#!/bin/bash

export LITELLM_MASTER_KEY="admin"
export DATABASE_URL="sqlite:///litellm.db"
litellm --config litellm_config.yaml --port 4000