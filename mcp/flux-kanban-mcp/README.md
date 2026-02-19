# Flux

https://github.com/sirsjg/flux

## Add to claude code

Run the install script:
```
./install-mcp-claude.sh
```

This will:
1. Pull the latest flux image
2. Start the flux-web container (port 3000)
3. Register the MCP server in Claude Code

Flux web UI: http://localhost:3000

## Manual setup

Start the web server:
```
./flux.sh
```

Register the MCP server:
```
claude mcp add flux -- podman run -i --userns=keep-id --rm -v flux-data:/app/packages/data -v flux-blobs:/home/flux -e FLUX_DATA=/app/packages/data/flux.sqlite sirsjg/flux-mcp:latest bun packages/mcp/dist/index.js
```

## Result

