# Open Memory

https://github.com/CaviraOSS/OpenMemory

🧠 Real long-term memory (not just embeddings in a table) <br/>
💾 Self-hosted, local-first (SQLite / Postgres) <br/>
🐍 Python + 🟦 Node SDKs <br/>
🧩 Integrations: LangChain, CrewAI, AutoGen, Streamlit, MCP, VS Code <br/>
📥 Sources: GitHub, Notion, Google Drive, OneDrive, Web Crawler <br/>
🔍 Explainable traces (see why something was recalled) <br/>

## Result

```
❯ ./run.sh
Added memory: {'root_memory_id': 'f91c96a1-5c12-4bd3-9026-a912a0e9fe03', 'child_count': 0, 'total_tokens': 6, 'strategy': 'single', 'extraction': {'content_type': 'text', 'char_count': 22, 'estimated_tokens': 6, 'extraction_method': 'passthrough'}, 'id': 'f91c96a1-5c12-4bd3-9026-a912a0e9fe03'}
Search results: [{'id': 'f91c96a1-5c12-4bd3-9026-a912a0e9fe03', 'content': 'user prefers dark mode', 'score': 0.5879796098811148, 'primary_sector': 'semantic', 'path': ['f91c96a1-5c12-4bd3-9026-a912a0e9fe03'], 'salience': 0.39999999995062935, 'last_seen_at': 1771225550440, 'tags': [], 'metadata': {'content_type': 'text', 'char_count': 22, 'estimated_tokens': 6, 'extraction_method': 'passthrough', 'ingestion_strategy': 'single', 'ingested_at': 1771225550440}}]
Deleted memory: f91c96a1-5c12-4bd3-9026-a912a0e9fe03
```