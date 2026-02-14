#!/bin/bash
podman exec -it twitter-postgres psql -U twitter -d twitter
