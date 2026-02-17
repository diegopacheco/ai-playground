# Oh My Claude Code

https://github.com/Yeachan-Heo/oh-my-claudecode

Team mode is on.

~/.claude/settings.json
```
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

## Result

```
❯ /oh-my-claudecode:team 3:executor "build a twitter clone like application with rust and react - have a run.sh"
```

```
✳ Building Rust backend API… (2m 15s · 187.3k tokens)
  ⎿  ✔ Build React frontend in frontend/ directory (@worker-2)
     ◼ Build Rust backend API in backend/ directory (@worker-1)
       Running Build Rust backend to verify compilation…
     ✔ Create run.sh and project root files (@worker-3)
```

<img src="result.png" width="600" />

## Experience Notes

* I did not got impressed with the teams feature. 
* I Saw 3 wrokers running in parallel, but the result was not good.
* There was a constant try to get rust in edition 2021 (instead if 2024)
* React code is all flat on the src folder.
* I did not like the code.
* Again the twitter app is pretty based just timeline and post did not have a lot of features like: follow, search, login, register, profile.