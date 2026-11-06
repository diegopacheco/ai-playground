# Changelog

Records user visible changes to the Personal Agent Protocol (Poppy) POC.

## 0.1.0 - 2026-11-05

### Added

- Willow Inn serves a Poppy draft 0.4 discovery document at `/.well-known/poppy.json`, an excluded tasks document and an OpenAPI document.
- Session routes accept only requests with `Poppy-Version: 0.4` and a valid ES256 `Poppy-Session` JWT whose delegation is signed by a key the Personal Agent published; replayed, stale, retargeted or foreign tokens get `invalid_session`.
- Rooms API to search rooms by price and pets and to hold a room for 1 to 14 nights, with holds visible only to the pairwise User ID that made them.
- Company Agent conversations: start, send, read events after a cursor and close, with idempotent message IDs.
- Feedback endpoint for tasks a Personal Agent could not do.
- `poppy.json` inspector that reports pass, warn or fail for each draft rule.
- Atlas Personal Agent UI with Discover, Rooms API, Company Agent, Feedback, Inspector and Wire log tabs, and an "Act for another user" button that switches the pairwise User ID.
- `setup.sh`, `start-all.sh`, `stop-all.sh`, `status.sh`, `test-all.sh` and `ui.sh` scripts, with ports in `scripts/ports.env`.

### Verified

- `./scripts/test-all.sh`: ty clean, 26 backend tests passed, tsc clean, 6 frontend tests passed, 0 skipped.
- `bun run build` produced a production bundle with no errors.
- In a real browser every tab worked against the running Company: browser signed Session JWTs were accepted, a River Suite hold for 2 nights showed a $640 total, the Company Agent answered with cursor reads, and feedback returned 202.
- Against the running Company, an unsigned request returned 401 `invalid_session` and a replayed token returned 401 "jti was already used".
- `start-all.sh` and `stop-all.sh` were each run twice; both ports went UP and then DOWN.
