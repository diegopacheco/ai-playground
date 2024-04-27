#!/bin/bash

podman run --rm -p 8080:8080 --name local-ai -ti localai/localai:latest-aio-cpu

