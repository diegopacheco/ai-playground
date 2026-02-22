# Moltis

https://www.moltis.org/

## Result

Motis Intalation <br/>
<img src="Moltis-Install.png" width="600" />

Motis Summary <br/>
<img src="Moltis-summary.png" width="600" />

Motis Chats <br/>
<img src="Moltis-Chat.png" width="600" />

## Experience Notes

* I had a separated VM with Ubuntu 24.03 LTS and created degicated email, and slack for complete isolateion and sandboxing.
* I Call the VM/OS Sebastian, I tought was necessary.
* Installation is very easy - was done in less than 1 min.
* UI It's simple and not complex.
* Only saw telegran (whish they had slack).
* Fast and Clean
* I like it

## Install

```bash
~ curl -fsSL https://www.moltis.org/install.sh | sh

  Moltis Installer
  Personal AI gateway - one binary, multiple LLM providers

==> Detected: linux (aarch64)
==> Fetching latest version...
==> Version: 0.9.10
==> Downloading moltis_0.9.10-1_arm64.deb...
==> Installing .deb package (requires sudo)...
[sudo] password for sebastian:
Selecting previously unselected package moltis.
(Reading database ... 173518 files and directories currently installed.)
Preparing to unpack .../moltis_0.9.10-1_arm64.deb ...
Unpacking moltis (0.9.10-1) ...
Setting up moltis (0.9.10-1) ...
==> Moltis installed via .deb package

==> Installation complete!
  moltis 0.9.10

Get started:
  moltis          # Start the gateway
  moltis --help   # Show help

Documentation: https://www.moltis.org/
```

### Run

```
➜  ~ moltis
2026-02-21T10:03:57.226566Z  INFO moltis: moltis starting version="0.9.10"
2026-02-21T10:03:57.227842Z  INFO moltis_config::loader: wrote default config template path=/home/sebastian/.config/moltis/moltis.toml
2026-02-21T10:03:57.242675Z  WARN moltis_agents::providers: failed to fetch live models for provider provider="ollama" error=error sending request for url (http://127.0.0.1:11434/api/tags)
2026-02-21T10:03:57.339932Z  INFO moltis_agents::providers::local_gguf: local-llm system info total_ram_gb=9 available_ram_gb=6 has_metal=false has_cuda=false tier=small (8GB)
2026-02-21T10:03:57.339968Z  INFO moltis_agents::providers::local_gguf: local-llm model cache directory cache_dir=/home/sebastian/.moltis/models
2026-02-21T10:03:57.339972Z  INFO moltis_agents::providers::local_gguf: suggested local model for your system model="deepseek-coder-6.7b-q4_k_m" display_name="DeepSeek Coder 6.7B (Q4_K_M)" min_ram_gb=8 backend=GGUF
2026-02-21T10:03:57.339997Z  INFO moltis_agents::providers::local_gguf: cached local models in model cache directory cached_models=[] cached_count=0
2026-02-21T10:03:57.340008Z  INFO moltis_agents::providers: local-llm enabled but no models configured. Add [providers.local] models = ["..."] to config.
2026-02-21T10:03:57.340020Z  INFO moltis_gateway::server: startup model inventory model_count=0 provider_count=0 provider_model_counts=[] sample_model_ids=[]
2026-02-21T10:03:57.340055Z  WARN moltis_gateway::server: no LLM providers at startup; model/chat services remain active and will pick up providers after credentials are saved provider_summary=no LLM providers configured config_path=/home/sebastian/.config/moltis/moltis.toml provider_keys_path=/home/sebastian/.config/moltis/provider_keys.json

⚠️  Browser tool enabled but no compatible browser was found!
No Chromium-based browser found. Install one:

  Debian/Ubuntu: sudo apt install chromium-browser
  Fedora:         sudo dnf install chromium
  Arch:           sudo pacman -S chromium
  # Alternatives: brave-browser, microsoft-edge-stable

Any Chromium-based browser works (Chrome, Chromium, Edge, Brave, Opera, Vivaldi).

Or set the path manually:
  [tools.browser]
  chrome_path = "/path/to/browser"

Or set the CHROME environment variable.

2026-02-21T10:03:57.340408Z  WARN moltis_browser::detect: Browser tool enabled but no compatible browser was found.
No Chromium-based browser found. Install one:

  Debian/Ubuntu: sudo apt install chromium-browser
  Fedora:         sudo dnf install chromium
  Arch:           sudo pacman -S chromium
  # Alternatives: brave-browser, microsoft-edge-stable

Any Chromium-based browser works (Chrome, Chromium, Edge, Brave, Opera, Vivaldi).

Or set the path manually:
  [tools.browser]
  chrome_path = "/path/to/browser"

Or set the CHROME environment variable.
2026-02-21T10:03:57.340684Z  INFO moltis_browser::manager: browser manager initialized (sandbox mode controlled per-session) sandbox_image=browserless/chrome
2026-02-21T10:03:57.340778Z  INFO moltis_gateway::server: startup configuration storage diagnostics user=sebastian home=/home/sebastian config_dir=/home/sebastian/.config/moltis discovered_config=/home/sebastian/.config/moltis/moltis.toml expected_config=/home/sebastian/.config/moltis/moltis.toml provider_keys_path=/home/sebastian/.config/moltis/provider_keys.json
2026-02-21T10:03:57.340829Z  INFO moltis_gateway::server: startup path diagnostics kind="config-dir" path=/home/sebastian/.config/moltis exists=true is_dir=true readonly=false size_bytes=4096
2026-02-21T10:03:57.340887Z  INFO moltis_gateway::server: startup write probe succeeded for config directory path=/home/sebastian/.config/moltis/.moltis-write-check-21161-1771668237340837101.tmp
2026-02-21T10:03:57.340917Z  INFO moltis_gateway::server: startup path diagnostics kind="config-file" path=/home/sebastian/.config/moltis/moltis.toml exists=true is_dir=false readonly=false size_bytes=34252
2026-02-21T10:03:57.340923Z  INFO moltis_gateway::server: provider key store file not found yet; it will be created after the first providers.save_key path=/home/sebastian/.config/moltis/provider_keys.json
2026-02-21T10:03:57.402979Z  INFO moltis_gateway::server: WebAuthn RP registered rp_id=localhost origins=["https://localhost:33959", "https://moltis.localhost:33959"]
2026-02-21T10:03:57.403045Z  INFO moltis_gateway::server: WebAuthn RP registered rp_id=ocean.local origins=["https://ocean.local:33959"]
2026-02-21T10:03:57.403056Z  INFO moltis_gateway::server: WebAuthn RP registered rp_id=ocean origins=["https://ocean:33959"]
2026-02-21T10:03:57.403060Z  INFO moltis_gateway::server: WebAuthn passkeys enabled origins=["https://localhost:33959", "https://moltis.localhost:33959", "https://ocean.local:33959", "https://ocean:33959"]
2026-02-21T10:03:57.403478Z  WARN moltis_tools::sandbox: no usable container runtime found; sandboxed execution will use direct host access
2026-02-21T10:03:57.411001Z  INFO moltis_gateway::server: 0 stored channel(s) found in database
2026-02-21T10:03:57.411293Z  INFO moltis_common::hooks: hook handler registered handler="boot-md"
2026-02-21T10:03:57.411307Z  INFO moltis_common::hooks: hook handler registered handler="command-logger"
2026-02-21T10:03:57.411310Z  INFO moltis_common::hooks: hook handler registered handler="session-memory"
2026-02-21T10:03:57.411323Z  INFO moltis_gateway::server: 4 hook(s) discovered (1 shell, 3 built-in), 3 registered
2026-02-21T10:03:57.415130Z  INFO moltis_gateway::server: memory: no embedding provider found, using keyword-only search
2026-02-21T10:03:57.425105Z  INFO moltis_gateway::server: memory system initialized embeddings=false
2026-02-21T10:03:57.425969Z  INFO moltis_gateway::server: memory: initial sync complete updated=0 unchanged=0 removed=0 errors=0 cache_hits=0 cache_misses=0
2026-02-21T10:03:57.426306Z  INFO moltis_gateway::server: memory: status files=0 chunks=0 db_size=52.0 KB model=none (keyword-only)
2026-02-21T10:03:57.426493Z  INFO moltis_metrics::recorder: Prometheus metrics exporter initialized
2026-02-21T10:03:57.426511Z  INFO moltis_gateway::server: Metrics collection enabled
2026-02-21T10:03:57.426668Z  INFO moltis_memory::watcher: file watcher: watching directory dir=/home/sebastian/.moltis
2026-02-21T10:03:57.426690Z  INFO moltis_gateway::server: memory: file watcher started
2026-02-21T10:03:57.433494Z  INFO moltis_gateway::server: Metrics history store initialized at /home/sebastian/.moltis/metrics.db
2026-02-21T10:03:57.433754Z  INFO moltis_browser::manager: browser manager initialized (sandbox mode controlled per-session) sandbox_image=browserless/chrome
2026-02-21T10:03:57.433873Z  INFO moltis_gateway::server: agent tools registered tools=["create_skill", "cron", "process", "transcribe", "exec", "calc", "browser", "memory_save", "session_state", "update_skill", "branch_session", "memory_get", "show_map", "speak", "memory_search", "web_fetch", "sandbox_packages", "delete_skill", "get_user_location"]
2026-02-21T10:03:57.434889Z  INFO moltis_skills::watcher: skill watcher: watching directory dir=/home/sebastian/.moltis/skills
2026-02-21T10:03:57.435480Z  INFO moltis_gateway::push: Generating new VAPID keys for push notifications
2026-02-21T10:03:57.435799Z  INFO moltis_gateway::push: VAPID keys generated and saved
2026-02-21T10:03:57.435818Z  INFO moltis_gateway::server: push notification service initialized
2026-02-21T10:03:57.436285Z  INFO moltis_gateway::tls: generating TLS certificates
2026-02-21T10:03:57.436482Z  INFO moltis_gateway::tls: certificates written dir=/home/sebastian/.config/moltis/certs
2026-02-21T10:03:57.436835Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/tmux/SKILL.md source=Personal name=tmux
2026-02-21T10:03:57.436869Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/template-skill/SKILL.md source=Personal name=template-skill
2026-02-21T10:03:57.439033Z  INFO moltis_gateway::server: ┌─────────────────────────────────────────────────────────────────────────────┐
2026-02-21T10:03:57.439052Z  INFO moltis_gateway::server: │  moltis gateway v0.9.10                                                     │
2026-02-21T10:03:57.439056Z  INFO moltis_gateway::server: │  protocol v3, listening on https://localhost:33959 (HTTP/2 + HTTP/1.1)      │
2026-02-21T10:03:57.439058Z  INFO moltis_gateway::server: │  bind (--bind): 127.0.0.1:33959                                             │
2026-02-21T10:03:57.439060Z  INFO moltis_gateway::server: │  193 methods registered                                                     │
2026-02-21T10:03:57.439081Z  INFO moltis_gateway::server: │  llm: no LLM providers configured                                           │
2026-02-21T10:03:57.439097Z  INFO moltis_gateway::server: │  skills: 2 enabled, 0 repos                                                 │
2026-02-21T10:03:57.439100Z  INFO moltis_gateway::server: │  mcp: 0 configured                                                          │
2026-02-21T10:03:57.439102Z  INFO moltis_gateway::server: │  sandbox: none backend                                                      │
2026-02-21T10:03:57.439105Z  INFO moltis_gateway::server: │  config: /home/sebastian/.config/moltis/moltis.toml                         │
2026-02-21T10:03:57.439107Z  INFO moltis_gateway::server: │  data: /home/sebastian/.moltis                                              │
2026-02-21T10:03:57.439123Z  INFO moltis_gateway::server: │  passkey origin: https://localhost:33959                                    │
2026-02-21T10:03:57.439125Z  INFO moltis_gateway::server: │  passkey origin: https://moltis.localhost:33959                             │
2026-02-21T10:03:57.439127Z  INFO moltis_gateway::server: │  passkey origin: https://ocean.local:33959                                  │
2026-02-21T10:03:57.439129Z  INFO moltis_gateway::server: │  passkey origin: https://ocean:33959                                        │
2026-02-21T10:03:57.439131Z  INFO moltis_gateway::server: │  ⚠ no container runtime found; installing packages on host in background    │
2026-02-21T10:03:57.439133Z  INFO moltis_gateway::server: │                                                                             │
2026-02-21T10:03:57.439136Z  INFO moltis_gateway::server: │  setup code: 461032                                                         │
2026-02-21T10:03:57.439138Z  INFO moltis_gateway::server: │  enter this code to set your password or register a passkey                 │
2026-02-21T10:03:57.439154Z  INFO moltis_gateway::server: │                                                                             │
2026-02-21T10:03:57.439355Z  INFO moltis_gateway::server: │  CA cert: http://localhost:33960/certs/ca.pem                               │
2026-02-21T10:03:57.439391Z  INFO moltis_gateway::server: │    or: /home/sebastian/.config/moltis/certs/ca.pem                          │
2026-02-21T10:03:57.439394Z  INFO moltis_gateway::server: │  run `moltis trust-ca` to remove browser warnings                           │
2026-02-21T10:03:57.439396Z  INFO moltis_gateway::server: └─────────────────────────────────────────────────────────────────────────────┘
2026-02-21T10:03:57.439724Z  INFO moltis_cron::service: loaded cron jobs count=0
2026-02-21T10:03:57.439758Z  INFO moltis_gateway::server: heartbeat skipped: no prompt in config and HEARTBEAT.md is empty
2026-02-21T10:03:57.439833Z  INFO moltis_gateway::tls: HTTP redirect server listening http://localhost:33960/certs/ca.pem
2026-02-21T10:03:57.440047Z  INFO moltis_gateway::server: Loaded 0 historical metrics points from store
2026-02-21T10:04:00.355806Z  WARN moltis_tools::sandbox: apt-get update failed (non-fatal) stderr=E: Release file for http://ports.ubuntu.com/ubuntu-ports/dists/noble-updates/InRelease is not valid yet (invalid for another 12h 18min 9s). Updates for this repository will not be applied.
E: Release file for http://ports.ubuntu.com/ubuntu-ports/dists/noble-backports/InRelease is not valid yet (invalid for another 12h 20min 28s). Updates for this repository will not be applied.
E: Release file for http://ports.ubuntu.com/ubuntu-ports/dists/noble-security/InRelease is not valid yet (invalid for another 12h 16min 31s). Updates for this repository will not be applied.

2026-02-21T10:04:00.599024Z  INFO moltis_tools::sandbox: provisioning host packages packages=autoconf automake bison chromium clang dnsutils flex gnupg2 libclang-dev liblzma-dev libsqlite3-dev libssl-dev libtool libxss1 libyaml-dev llvm-dev lz4 net-tools npm p7zip-full patchelf pigz pkg-config python-is-python3 python3-dev python3-pip python3-venv ripgrep ruby ruby-dev shellcheck sqlite3 tree sudo=true
2026-02-21T10:04:00.818058Z  WARN moltis_tools::sandbox: apt-get install failed (non-fatal) stderr=E: Unable to correct problems, you have held broken packages.

2026-02-21T10:04:00.818090Z  INFO moltis_gateway::server: host package provisioning complete installed=0 skipped=39 sudo=true
2026-02-21T10:05:26.659776Z  INFO moltis_gateway::server: Serving assets from embedded binary
2026-02-21T10:05:26.660927Z  INFO moltis_gateway::chat: models.list response model_count=0
2026-02-21T10:05:27.003823Z  INFO moltis_gateway::ws: ws: new connection conn_id=55fe4594-4f00-49b3-8bf6-07dbd9b8b575 remote_ip=127.0.0.1
2026-02-21T10:05:27.008008Z  INFO moltis_gateway::ws: ws: handshake complete conn_id=55fe4594-4f00-49b3-8bf6-07dbd9b8b575 client_id=web-chat-ui client_version=0.1.0 role=operator
2026-02-21T10:05:27.008119Z  INFO moltis_gateway::ws: ws: auto-persisted browser timezone to USER.md conn_id=55fe4594-4f00-49b3-8bf6-07dbd9b8b575 timezone=America/Los_Angeles
2026-02-21T10:06:21.997460Z  INFO moltis_gateway::provider_setup: provider setup operation started operation="providers.validate_key" provider=openai
2026-02-21T10:06:22.603901Z  INFO moltis_agents::providers::local_gguf: local-llm system info total_ram_gb=9 available_ram_gb=6 has_metal=false has_cuda=false tier=small (8GB)
2026-02-21T10:06:22.603956Z  INFO moltis_agents::providers::local_gguf: local-llm model cache directory cache_dir=/home/sebastian/.moltis/models
2026-02-21T10:06:22.603961Z  INFO moltis_agents::providers::local_gguf: suggested local model for your system model="deepseek-coder-6.7b-q4_k_m" display_name="DeepSeek Coder 6.7B (Q4_K_M)" min_ram_gb=8 backend=GGUF
2026-02-21T10:06:22.603979Z  INFO moltis_agents::providers::local_gguf: cached local models in model cache directory cached_models=[] cached_count=0
2026-02-21T10:06:22.603984Z  INFO moltis_agents::providers: local-llm enabled but no models configured. Add [providers.local] models = ["..."] to config.
2026-02-21T10:06:22.603999Z  INFO moltis_gateway::provider_setup: provider validation discovered candidate models for probing provider=openai model_count=71
2026-02-21T10:06:22.604011Z  INFO moltis_gateway::provider_setup: provider validation model probe started provider=openai model=openai::gpt-5.2-codex attempt=1 total_models=71
2026-02-21T10:06:23.663488Z  INFO moltis_gateway::provider_setup: provider validation model probe failed provider=openai model=openai::gpt-5.2-codex elapsed_ms=1059 unsupported=true
2026-02-21T10:06:23.663541Z  INFO moltis_gateway::provider_setup: provider validation model probe started provider=openai model=openai::gpt-5.2-chat-latest attempt=2 total_models=71
2026-02-21T10:06:25.290858Z  INFO moltis_gateway::provider_setup: provider validation model probe succeeded provider=openai model=openai::gpt-5.2-chat-latest elapsed_ms=1627
2026-02-21T10:06:25.291067Z  INFO moltis_gateway::provider_setup: provider setup operation finished operation="providers.validate_key" provider=openai elapsed_ms=3293
2026-02-21T10:06:25.292789Z  INFO moltis_gateway::provider_setup: provider setup operation started operation="providers.save_key" provider=openai
2026-02-21T10:06:25.292816Z  INFO moltis_gateway::provider_setup: saving provider config provider="openai" has_api_key=true has_base_url=false models=0 key_store_path=/home/sebastian/.config/moltis/provider_keys.json
2026-02-21T10:06:25.821301Z  WARN moltis_agents::providers: failed to fetch live models for provider provider="ollama" error=error sending request for url (http://127.0.0.1:11434/api/tags)
2026-02-21T10:06:25.860822Z  INFO moltis_agents::providers::local_gguf: local-llm system info total_ram_gb=9 available_ram_gb=6 has_metal=false has_cuda=false tier=small (8GB)
2026-02-21T10:06:25.860863Z  INFO moltis_agents::providers::local_gguf: local-llm model cache directory cache_dir=/home/sebastian/.moltis/models
2026-02-21T10:06:25.860876Z  INFO moltis_agents::providers::local_gguf: suggested local model for your system model="deepseek-coder-6.7b-q4_k_m" display_name="DeepSeek Coder 6.7B (Q4_K_M)" min_ram_gb=8 backend=GGUF
2026-02-21T10:06:25.860896Z  INFO moltis_agents::providers::local_gguf: cached local models in model cache directory cached_models=[] cached_count=0
2026-02-21T10:06:25.860908Z  INFO moltis_agents::providers: local-llm enabled but no models configured. Add [providers.local] models = ["..."] to config.
2026-02-21T10:06:25.860917Z  INFO moltis_gateway::provider_setup: saved provider config to disk and rebuilt provider registry provider="openai" provider_summary=1 provider, 72 models models=72
2026-02-21T10:06:25.860940Z  INFO moltis_gateway::provider_setup: provider setup operation finished operation="providers.save_key" provider=openai elapsed_ms=568
2026-02-21T10:06:40.020647Z  INFO moltis_gateway::chat: model probe started model_id="openai::gpt-5.1-codex-mini" provider="openai"
2026-02-21T10:06:40.173768Z  WARN moltis_gateway::chat: model probe failed model_id="openai::gpt-5.1-codex-mini" provider="openai" elapsed_ms=153 error=This model is only supported in v1/responses and not in v1/chat/completions.
2026-02-21T10:06:53.182050Z  INFO moltis_gateway::provider_setup: provider setup operation started operation="providers.save_key" provider=openai
2026-02-21T10:06:53.182093Z  INFO moltis_gateway::provider_setup: saving provider config provider="openai" has_api_key=true has_base_url=false models=0 key_store_path=/home/sebastian/.config/moltis/provider_keys.json
2026-02-21T10:06:53.648487Z  WARN moltis_agents::providers: failed to fetch live models for provider provider="ollama" error=error sending request for url (http://127.0.0.1:11434/api/tags)
2026-02-21T10:06:53.690436Z  INFO moltis_agents::providers::local_gguf: local-llm system info total_ram_gb=9 available_ram_gb=6 has_metal=false has_cuda=false tier=small (8GB)
2026-02-21T10:06:53.690475Z  INFO moltis_agents::providers::local_gguf: local-llm model cache directory cache_dir=/home/sebastian/.moltis/models
2026-02-21T10:06:53.690481Z  INFO moltis_agents::providers::local_gguf: suggested local model for your system model="deepseek-coder-6.7b-q4_k_m" display_name="DeepSeek Coder 6.7B (Q4_K_M)" min_ram_gb=8 backend=GGUF
2026-02-21T10:06:53.690499Z  INFO moltis_agents::providers::local_gguf: cached local models in model cache directory cached_models=[] cached_count=0
2026-02-21T10:06:53.690510Z  INFO moltis_agents::providers: local-llm enabled but no models configured. Add [providers.local] models = ["..."] to config.
2026-02-21T10:06:53.690533Z  INFO moltis_gateway::provider_setup: saved provider config to disk and rebuilt provider registry provider="openai" provider_summary=1 provider, 72 models models=72
2026-02-21T10:06:53.690540Z  INFO moltis_gateway::provider_setup: provider setup operation finished operation="providers.save_key" provider=openai elapsed_ms=508
2026-02-21T10:06:53.691520Z  INFO moltis_gateway::provider_setup: provider setup operation started operation="providers.save_models" provider=openai
2026-02-21T10:06:53.691667Z  INFO moltis_gateway::provider_setup: saved model preferences and queued async registry rebuild provider="openai" count=1 models=["openai::gpt-5.1-codex-mini"]
2026-02-21T10:06:53.691744Z  INFO moltis_gateway::provider_setup: provider setup operation finished operation="providers.save_models" provider=openai elapsed_ms=0
2026-02-21T10:06:53.691990Z  INFO moltis_gateway::provider_setup: provider registry async rebuild started provider=openai reason="save_models" rebuild_seq=1
2026-02-21T10:06:54.192357Z  WARN moltis_agents::providers: failed to fetch live models for provider provider="ollama" error=error sending request for url (http://127.0.0.1:11434/api/tags)
2026-02-21T10:06:54.232175Z  INFO moltis_agents::providers::local_gguf: local-llm system info total_ram_gb=9 available_ram_gb=6 has_metal=false has_cuda=false tier=small (8GB)
2026-02-21T10:06:54.232368Z  INFO moltis_agents::providers::local_gguf: local-llm model cache directory cache_dir=/home/sebastian/.moltis/models
2026-02-21T10:06:54.232374Z  INFO moltis_agents::providers::local_gguf: suggested local model for your system model="deepseek-coder-6.7b-q4_k_m" display_name="DeepSeek Coder 6.7B (Q4_K_M)" min_ram_gb=8 backend=GGUF
2026-02-21T10:06:54.232394Z  INFO moltis_agents::providers::local_gguf: cached local models in model cache directory cached_models=[] cached_count=0
2026-02-21T10:06:54.232399Z  INFO moltis_agents::providers: local-llm enabled but no models configured. Add [providers.local] models = ["..."] to config.
2026-02-21T10:06:54.232510Z  INFO moltis_gateway::provider_setup: provider registry async rebuild finished provider=openai reason="save_models" rebuild_seq=1 provider_summary=1 provider, 72 models models=72 elapsed_ms=540
2026-02-21T10:08:35.055620Z  INFO moltis_gateway::chat: models.list response model_count=51
2026-02-21T10:08:35.056671Z  INFO moltis_gateway::chat: models.list response model_count=51
2026-02-21T10:08:35.106057Z  INFO moltis_gateway::chat: models.list response model_count=51
2026-02-21T10:08:35.108443Z  INFO moltis_gateway::chat: models.list response model_count=51
2026-02-21T10:09:02.781776Z  INFO moltis_gateway::ws: ws: connection closed conn_id=55fe4594-4f00-49b3-8bf6-07dbd9b8b575 duration_secs=215
2026-02-21T10:09:02.789615Z  INFO moltis_gateway::chat: models.list response model_count=51
2026-02-21T10:09:03.036803Z  INFO moltis_gateway::chat: models.list response model_count=51
2026-02-21T10:09:03.037733Z  INFO moltis_gateway::chat: models.list response model_count=51
2026-02-21T10:09:03.042784Z  INFO moltis_gateway::ws: ws: new connection conn_id=43588fff-5d9f-450b-83e5-97d116579ea2 remote_ip=127.0.0.1
2026-02-21T10:09:03.047873Z  INFO moltis_gateway::ws: ws: handshake complete conn_id=43588fff-5d9f-450b-83e5-97d116579ea2 client_id=web-chat-ui client_version=0.1.0 role=operator
2026-02-21T10:09:03.049829Z  INFO moltis_gateway::chat: models.list response model_count=51
2026-02-21T10:09:03.072050Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/tmux/SKILL.md source=Personal name=tmux
2026-02-21T10:09:03.072119Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/template-skill/SKILL.md source=Personal name=template-skill
2026-02-21T10:09:13.934594Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/tmux/SKILL.md source=Personal name=tmux
2026-02-21T10:09:13.934643Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/template-skill/SKILL.md source=Personal name=template-skill
2026-02-21T10:09:13.939962Z  INFO moltis_gateway::chat: chat.send run_id=85c70ab8-f5f5-43be-ae49-2af1442cad15 user_message=hi what can you do ? model="openai::gpt-5.3" stream_only=false session=main reply_medium=Text client_seq=Some(1)
2026-02-21T10:09:13.946113Z  INFO moltis_agents::runner: starting streaming agent loop provider="openai" model="openai::gpt-5.3" native_tools=true tools_count=19 is_multimodal=false
2026-02-21T10:09:13.946301Z  INFO moltis_agents::runner: schemas_for_api prepared for streaming native_tools=true schemas_for_api_count=19 tool_schemas_count=19
2026-02-21T10:09:13.946318Z  INFO moltis_agents::runner: calling LLM (streaming) iteration=1 messages_count=2
2026-02-21T10:09:14.038962Z  WARN moltis_gateway::chat: agent run error run_id="85c70ab8-f5f5-43be-ae49-2af1442cad15" error=HTTP 404: {
    "error": {
        "message": "The model `gpt-5.3` does not exist or you do not have access to it.",
        "type": "invalid_request_error",
        "param": null,
        "code": "model_not_found"
    }
}

2026-02-21T10:09:17.794351Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/tmux/SKILL.md source=Personal name=tmux
2026-02-21T10:09:17.794409Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/template-skill/SKILL.md source=Personal name=template-skill
2026-02-21T10:09:17.808791Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/tmux/SKILL.md source=Personal name=tmux
2026-02-21T10:09:17.808825Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/template-skill/SKILL.md source=Personal name=template-skill
2026-02-21T10:09:29.901634Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/tmux/SKILL.md source=Personal name=tmux
2026-02-21T10:09:29.901688Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/template-skill/SKILL.md source=Personal name=template-skill
2026-02-21T10:09:29.907337Z  INFO moltis_gateway::chat: chat.send run_id=a70931fd-5e36-411e-9b3e-34224b70f17d user_message=hi what can you do? model="openai::gpt-5.1" stream_only=false session=main reply_medium=Text client_seq=Some(2)
2026-02-21T10:09:29.908774Z  INFO moltis_agents::runner: starting streaming agent loop provider="openai" model="openai::gpt-5.1" native_tools=true tools_count=19 is_multimodal=false
2026-02-21T10:09:29.908864Z  INFO moltis_agents::runner: schemas_for_api prepared for streaming native_tools=true schemas_for_api_count=19 tool_schemas_count=19
2026-02-21T10:09:29.908872Z  INFO moltis_agents::runner: calling LLM (streaming) iteration=1 messages_count=3
2026-02-21T10:09:37.653160Z  INFO moltis_agents::runner: streaming LLM response complete iteration=1 has_text=true tool_calls_count=0 input_tokens=5171 output_tokens=400
2026-02-21T10:09:37.653300Z  INFO moltis_agents::runner: streaming agent loop complete — returning text iterations=1 tool_calls=0
2026-02-21T10:09:37.653377Z  INFO moltis_gateway::chat: agent run complete run_id="a70931fd-5e36-411e-9b3e-34224b70f17d" iterations=1 tool_calls=0 response=I’m moltis 🐙, your resident crab with Wi‑Fi.

Here’s the short version of what I can do for you:

1. **Explain & teach stuff**
   - Math, coding, science, history, finance, writing, you name it.
   - Break things down step‑by‑step or give quick summaries.

2. **Write & edit**
   - Emails, essays, blog posts, resumes, cover letters, reports, documentation.
   - Rewrite for tone (more formal, more funny, simpler, more concise, etc.).
   - Fix grammar, structure, and clarity.

3. **Code & debug**
   - Help you write code (Python, JS/TS, Rust, Go, etc.).
   - Debug errors, explain stack traces, suggest better designs.
   - Sketch APIs, data models, small scripts or tools.

4. **Plan & organize**
   - Study plans, workout plans, project breakdowns, learning roadmaps.
   - Help you structure a big idea into tasks and milestones.

5. **Reason & solve problems**
   - Puzzles, logic problems, algorithm questions.
   - “What should I do about X?” type reasoning with pros/cons.

6. **Use tools**
   - Run shell commands in a sandbox (for experiments & small scripts).
   - Browse the web when you need live or specific info.
   - Set reminders via a scheduler.
   - Transcribe audio you upload.
   - Show maps if you’re asking about locations/directions.

7. **Be a thinking partner**
   - Brainstorm ideas, names, story plots, product concepts.
   - Roleplay scenarios (interviews, negotiations, conversations).

If you tell me what you’re trying to do *right now* (e.g., “help me write a resume”, “explain this bug”, “teach me X from scratch”), I’ll jump straight into that. silent=false
2026-02-21T10:09:37.653470Z  INFO moltis_gateway::chat: push: checking push notification (agent mode)
2026-02-21T10:09:37.653477Z  INFO moltis_gateway::chat: push notification skipped: no subscribers
2026-02-21T10:10:50.717795Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/tmux/SKILL.md source=Personal name=tmux
2026-02-21T10:10:50.717848Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/template-skill/SKILL.md source=Personal name=template-skill
2026-02-21T10:10:50.738705Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/tmux/SKILL.md source=Personal name=tmux
2026-02-21T10:10:50.738761Z  INFO moltis_skills::discover: loaded SKILL.md path=/home/sebastian/.moltis/skills/template-skill/SKILL.md source=Personal name=template-skill
```