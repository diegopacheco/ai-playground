# Ralph

https://github.com/snarktank/ralph

## First

```
 Load the prd skill and create a PRD for [your feature description] from @scripts/CLAUDE.md and @scripts/ralph.sh
```

Result:
```
⏺ Created all 3 files. Here's what was generated:

  scripts/prd.json - PRD for SnarkTank, a Twitter-like microblogging app with 10 user stories:

  ┌────────┬───────────────────────────────────┬──────────┐
  │   ID   │               Story               │ Priority │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-001 │ Project Setup and Database Schema │ 1        │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-002 │ User Registration and Login       │ 2        │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-003 │ Post a Snark                      │ 3        │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-004 │ Timeline Feed                     │ 4        │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-005 │ Like a Snark                      │ 5        │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-006 │ User Profile Page                 │ 6        │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-007 │ Follow and Unfollow Users         │ 7        │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-008 │ Following-Only Timeline           │ 8        │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-009 │ Reply to Snarks                   │ 9        │
  ├────────┼───────────────────────────────────┼──────────┤
  │ ST-010 │ Search Users and Snarks           │ 10       │
  └────────┴───────────────────────────────────┴──────────┘

  Tech stack: React + TypeScript frontend, Node.js + Express backend, SQLite database, plain CSS.

  Branch: ralph/snarktank-twitter

  scripts/progress.txt - Initialized with empty Codebase Patterns section.

  scripts/prompt.md - AMP-compatible prompt matching the CLAUDE.md instructions.

  To run Ralph: ./scripts/ralph.sh --tool claude 10
```

## Second - Raplh Loop

```
❯ ./scripts/ralph.sh --tool claude 10
Starting Ralph - Tool: claude - Max iterations: 10

===============================================================
  Ralph Iteration 1 of 10 (claude)
===============================================================
```