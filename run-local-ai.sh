#!/bin/bash

podman run -it --rm -v ~/localai/:/build/models:cached -p 8080:8080 --name local-ai -ti localai/localai:latest-aio-cpu
