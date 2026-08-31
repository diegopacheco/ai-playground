---
name: extract-ticket-fields
params: $ticket_body, $max_summary_words
---

Extract structured fields from the support ticket below.

Reply with a single JSON object and nothing else. No markdown fences, no commentary.

The object has exactly these keys:

"category": one of "billing", "bug", "feature_request", "account_access"
"severity": one of "low", "medium", "high"
"summary": the customer problem in at most $max_summary_words words

Ticket:
$ticket_body
