# city-zoning

A React + Vite (Bun) frontend over a zero-dependency Node backend that checks
building proposals against municipal zoning rules. Built as a target codebase
for the `agent-skill-linter` skill.

## Zoning rules

Five zones (R1, R3, C1, I1, MU). Each proposal is checked against:

- permitted uses for the zone
- maximum building height
- maximum lot coverage
- minimum lot area
- maximum floor area ratio (FAR)
- minimum front, side, and rear setbacks

A permit fee is computed for compliant proposals based on floor area, use, and height.

## Run

```
./start.sh
./stop.sh
```

- Backend: http://localhost:4000 (`/api/zones`, `POST /api/evaluate`)
- Frontend: http://localhost:5180

Both ports are strict: if either is already in use, `start.sh` stops with a clear
message instead of silently moving to another port. Override with
`SERVER_PORT` / `WEB_PORT`.

## Test

```
./test.sh
```
