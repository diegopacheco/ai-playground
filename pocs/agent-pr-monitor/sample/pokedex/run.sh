#!/bin/bash
cd "$(dirname "$0")"
npx ng build
echo "Serving on http://localhost:4200"
deno run --allow-net --allow-read server.ts
