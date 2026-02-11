#!/bin/bash
podman exec -it twitter_postgres psql -U twitter_user -d twitter_db
