# Changelog

Records the user-visible changes of the MCP server gateway router POC.

## 1.1.0 - 2026-11-06

### Added

- The gateway watches `MCP_DIR` (`dynamic-mcps/` in `start-all.sh`) and registers or drops servers while it runs, with no restart.
- Code servers are `tools.js` folders loaded in process. Markdown servers are `.md` files whose tools are written in plain English and carried out by `claude -p`.
- `GET /servers`, `POST /servers` (url, markdown or module) and `DELETE /servers/<name>`.
- Sample servers `dynamic-mcps/hash-mcp` (`sha256`) and `dynamic-mcps/writer.md` (`summarize`, `translate`).

### Verified

- `node --test` ran 27 tests three times: 27 passed, 0 failed, 0 skipped.
- On a live gateway with the real claude CLI, `writer__translate` returned "O gateway roteia todas as chamadas de ferramentas." in 3.3 s. A `live.md` written while the gateway ran was listed, and `live__shout` returned `ROUTER!`. Once the file was deleted, the tool was gone from the list.

## 1.0.0 - 2026-11-05

### Added

- math-mcp upstream MCP server on port 8101 with the `add` and `multiply` tools.
- text-mcp upstream MCP server on port 8102 with the `uppercase`, `reverse` and `word_count` tools.
- Gateway MCP server on port 8100 that lists the tools of every upstream as `<upstream>__<tool>` and routes each tool call to the upstream that owns it.
- Upstreams are configured with the `UPSTREAMS` env var as `name=url` pairs.
- An upstream that is down drops out of the gateway tool list, and calling one of its tools returns an error naming the unreachable URL.
- `GET /health` on all three servers.
- Scripts `setup.sh`, `start-all.sh`, `stop-all.sh`, `status.sh` and `test-all.sh`, with ports in `scripts/ports.env`.

### Verified

- `./scripts/test-all.sh` ran 16 tests: 16 passed, 0 failed, 0 skipped.
- With `./scripts/start-all.sh` running, `tools/list` on the gateway returned `math__add math__multiply text__uppercase text__reverse text__word_count`.
- `math__multiply` with 6 and 7 returned `42`, and `text__reverse` with `gateway` returned `yawetag`.
- `start-all.sh`, `stop-all.sh` and `setup.sh` ran twice in a row with exit 0, `status.sh` exited 0 from a nested folder, and all three ports went UP and then DOWN.
