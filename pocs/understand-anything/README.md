# Understand Anything

https://github.com/Lum1104/Understand-Anything

## Install

```
claude
/plugin marketplace add Lum1104/Understand-Anything
/plugin install understand-anything
```

## Experience Notes

```
* When I point neety - I got this: Netty has 3,511 Java files — far beyond the 100-file gate. How would you like to scope this?
* I told him to just focus on the netty-buffer module.
* Skill was running 4 agents in parallel to understand the codebase
* 
```

## Skill working

```
  Model: Opus 4.7 | Ctx: 90.2k | ⎇ main | (+0,-0)                                                                                     ⧉ In README.md
  3 local agents

❯ ⏺ main                                                                                                                 ↑/↓ to select · Enter to view
  ◯ understand-anything:fil…  Analyze batch 0 (netty-buffer)                                                                   2m 13s · ↓ 43.6k tokens
  ◯ understand-anything:fil…  Analyze batch 1 (netty-buffer)                                                                   2m 31s · ↓ 44.4k tokens
  ◯ understand-anything:fil…  Analyze batch 2 (netty-buffer)                                                                   2m 25s · ↓ 35.6k tokens
  ◯ understand-anything:fil…  Analyze batch 3 (netty-buffer)                                                                   2m 20s · ↓ 34.3k tokens
```