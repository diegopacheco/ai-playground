#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${CC_TOKEN_BAR_HOME:-$HOME/.cc-token-bar}"
EVENT="${1:?event required}"

mkdir -p "$DATA_DIR/sessions" "$DATA_DIR/tools" "$DATA_DIR/state"

write_atomic() {
  local target="$1"
  local content="$2"
  local tmp="${target}.tmp.$$"
  printf '%s' "$content" > "$tmp"
  mv "$tmp" "$target"
}

aggregate_transcript() {
  local sid="$1"
  local tpath="$2"
  local cwd="$3"
  local out="$DATA_DIR/sessions/$sid.json"

  local file_mtime
  file_mtime="$(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%SZ' "$tpath")"

  local agg
  agg="$(jq -s --arg sid "$sid" --arg p "$cwd" --arg fallback "$file_mtime" '
    def ts_list: [.[].timestamp // empty] | map(select(. != ""));
    {
      session_id: $sid,
      project_path: $p,
      started_at: (ts_list | (min // $fallback)),
      updated_at: (ts_list | (max // $fallback)),
      by_model: (
        reduce (.[] | select(.message? and .message.usage?)) as $m ({};
          .[$m.message.model] = (
            (.[$m.message.model] // {input_tokens:0,output_tokens:0,cache_creation_input_tokens:0,cache_read_input_tokens:0,web_search_requests:0,web_fetch_requests:0,messages:0})
            | .input_tokens += ($m.message.usage.input_tokens // 0)
            | .output_tokens += ($m.message.usage.output_tokens // 0)
            | .cache_creation_input_tokens += ($m.message.usage.cache_creation_input_tokens // 0)
            | .cache_read_input_tokens += ($m.message.usage.cache_read_input_tokens // 0)
            | .web_search_requests += ($m.message.usage.server_tool_use.web_search_requests // 0)
            | .web_fetch_requests  += ($m.message.usage.server_tool_use.web_fetch_requests // 0)
            | .messages += 1
          )
        )
      )
    }
  ' "$tpath")"
  write_atomic "$out" "$agg"
}

payload="$(cat)"

case "$EVENT" in
  post_tool_use)
    sid="$(jq -r '.session_id // empty' <<<"$payload")"
    [[ -n "$sid" ]] || exit 0
    tool="$(jq -r '.tool_name // empty' <<<"$payload")"
    [[ -n "$tool" ]] || exit 0
    in_b="$(jq -r '(.tool_input // {}) | tostring | length' <<<"$payload")"
    out_b="$(jq -r '(.tool_response // {}) | tostring | length' <<<"$payload")"
    f="$DATA_DIR/tools/$sid.json"
    if [[ -f "$f" ]]; then
      updated="$(jq --arg t "$tool" --argjson ib "$in_b" --argjson ob "$out_b" '
        .tools[$t] = ((.tools[$t] // {count:0,input_bytes:0,output_bytes:0})
          | .count += 1
          | .input_bytes += $ib
          | .output_bytes += $ob)
        | .updated_at = (now | todate)
      ' "$f")"
    else
      updated="$(jq -n --arg s "$sid" --arg t "$tool" --argjson ib "$in_b" --argjson ob "$out_b" '
        {session_id:$s, updated_at:(now|todate), tools:{($t):{count:1,input_bytes:$ib,output_bytes:$ob}}}
      ')"
    fi
    write_atomic "$f" "$updated"
    ;;

  stop|session_end)
    sid="$(jq -r '.session_id // empty' <<<"$payload")"
    [[ -n "$sid" ]] || exit 0
    tpath="$(jq -r '.transcript_path // empty' <<<"$payload")"
    [[ -n "$tpath" && -f "$tpath" ]] || exit 0
    cwd="$(jq -r '.cwd // ""' <<<"$payload")"
    aggregate_transcript "$sid" "$tpath" "$cwd"
    ;;

  backfill)
    sid="$(jq -r '.session_id' <<<"$payload")"
    tpath="$(jq -r '.transcript_path' <<<"$payload")"
    cwd="$(jq -r '.cwd' <<<"$payload")"
    aggregate_transcript "$sid" "$tpath" "$cwd"
    ;;
esac
exit 0