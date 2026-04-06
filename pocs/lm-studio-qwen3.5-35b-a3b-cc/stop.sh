#!/bin/bash

lsof -ti:5173 | xargs kill -9 2>/dev/null || true
