# Adversarial Review Results

## Round 1: Spec Review (Sarcasmotron)
25 flaws found. Key issues addressed:
- Timestamp format unspecified -> fixed to Unix seconds
- Cookie attributes missing -> added SameSite, HttpOnly, Path
- CORS omission -> added CORS config
- Admin seed display_name missing -> set to "Admin"
- Foreign key pragma scope -> added per-connection
- Search term sanitization undefined -> defined escaping rules

## Round 2: Implementation Review (Sarcasmotron)
10 critical flaws found. Fixes applied:

1. **Byte vs char length** - Changed all `.len()` to `.chars().count()` for character limits
2. **PRAGMA foreign_keys** - Added per-connection execution before init_db
3. **TOCTOU race in register** - Moved bcrypt outside mutex, use INSERT + UNIQUE error
4. **Bio missing control char check** - Added `.is_control()` validation
5. **Missing SameSite=Lax** - Added to all cookie builds
6. **Lock ordering in search** - Extract session check before db lock
7. **N+1 queries** - Acknowledged, acceptable for SQLite single-connection design
8. **Integration test body assertions** - Partial coverage (admin profile, hot topics verified)
9. **Frontend toggleLike error handling** - Added try/catch
10. **Frontend API Content-Type** - Only set when body present, handle non-JSON errors
