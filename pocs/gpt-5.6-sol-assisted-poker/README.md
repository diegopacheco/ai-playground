# Riverlight Poker School

Riverlight is a light-themed Texas Hold’em learning app built with Python 3.14 and Django 6. It combines a complete card reference, a guided hand trainer, and a heads-up table powered by local agent CLIs.

## What is inside

The Cards & Hands tab covers all 52 cards, suit behavior, the ten standard hand ranks, five-card frequencies, and the rule for choosing the best five cards from seven.

The Guided Hand tab deals private cards, reveals each street, estimates heads-up equity, names the current hand, and explains the recommended action at every decision point. A persistent four-street timeline keeps every recommendation and ends with a complete hand review.

The Agent Table tab lets the player select one of three local opponents:

- Claude through `claude -p`
- Codex through `codex exec`
- Agy through `agy -p`

The selected opponent is stored in the Django session. Agent play keeps assistance hidden until the player opens the optional probability path. That view shows win, tie, and loss rates, improvement routes, unseen cards, and the calculation method. If a CLI is unavailable, times out, or returns an invalid move, the table uses a local policy so the hand can continue.

## Requirements

- Python 3.14.x available as `python3.14`
- `curl`
- One or more supported agent CLIs for live agent decisions

No JavaScript or CSS libraries are used.

## Start

```bash
./start.sh
```

The script creates `.venv`, installs Django, applies migrations, and starts the server at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Stop

```bash
./stop.sh
```

## Test

```bash
./test.sh
```

The tests cover hand comparison, ace-low straights, repeatable equity estimates, all three pages, session persistence, guided street progression, and agent moves.

## Project layout

```text
academy/
  agents.py
  poker.py
  tests.py
  urls.py
  views.py
poker_school/
  settings.py
  urls.py
static/academy/
  app.js
  style.css
templates/academy/
  card.html
  home.html
manage.py
start.sh
stop.sh
test.sh
```

`academy/poker.py` owns the deck, hand evaluator, equity sampler, dealing flow, and teaching guidance. `academy/agents.py` owns the constrained CLI calls and local fallback policy. Django views store active hands and opponent choice in the database-backed session workflow.

## Agent behavior

Each agent receives the current street, visible board, pot, legal actions, and its estimated equity. It is instructed to return exactly one legal action. Calls use argument arrays without a shell and stop after 45 seconds.

The app is intended for learning and practice. Equity values are sampled estimates, not guarantees of a hand result.
