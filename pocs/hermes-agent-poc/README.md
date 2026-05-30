# Hermes Agent POC

## Install

```
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
hermes setup
```

## Experience Notes

* Setup takes a long time in: → Trying SSH clone... 
* Hermes has support for messaging like Slack
* Pretty Fast
* Goes directly to my api tokens on anthropic - try it with opus 4.7

## Results

Built a Rock Paper Scissors web app via Hermes Agent.

### Run

```
./start.sh    # serves on http://localhost:8000 and opens browser
./stop.sh     # stops the server
```

Optional: `PORT=9000 ./start.sh` to use a different port.

### Files

- `index.html` - the game (HTML + CSS + JS, single file)
- `start.sh`   - start local web server
- `stop.sh`    - stop local web server

### Screenshot

![Rock Paper Scissors gameplay](./screenshot.png)
