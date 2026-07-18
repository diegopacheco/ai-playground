#!/usr/bin/env bash
set -euo pipefail
ui=http://localhost:4321
api=http://localhost:8099
printf '%s\n' \
  "$ui/                      consoles" \
  "$ui/projects              projects and connections" \
  "$ui/audit-trail           audit trail" \
  "$ui/users                 user management" \
  "$ui/settings/ai           ai query authoring" \
  "$api/swagger              swagger ui" \
  "$api/v3/api-docs          openapi json" \
  "$api/actuator/health      backend health" \
  "http://localhost:6006/    storybook" \
  "" \
  "login                     admin / admin"
