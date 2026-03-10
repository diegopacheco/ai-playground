# /simplify

`/simplify` is a bundled Claude Code slash command that reviews changed code for reuse, quality, and efficiency, then automatically fixes any issues found.

## Experience Notes

* Claude Code is sucking memory like crazy, saw a warning for 2 or 3 GB used.
* simplify command uses a lot of sub-agents
* simplify uses a lot of tokens
* Claude seems to be focus on lines of code which works in some cases but is not the best.
* I wish was focused on better design and architecture, not just line count.
* I see that in a pojo it compact the lines to be all in one line, ugly and cheating if you asked me.

## What it does

- Reviews code for **reuse opportunities** (duplicated logic, patterns that could be consolidated)
- Assesses **code quality** (readability, maintainability, best practices)
- Checks **efficiency** (unnecessary complexity, performance improvements)
- **Fixes issues** automatically by applying simplifications directly to the code

## How to use

Type `/simplify` in Claude Code after making code changes. It analyzes your current changes and applies improvements.


## Result

Look [simplify-report.md](simplify-report.md) for the full before/after code snippets and explanations of each change.

```
⏺ Done. Here's what /simplify fixed:

665 lines -> 271 lines (59% reduction)

┌───────────────────────────────────────────┬─────────────────────────────────────────────┐
│                    Fix                    │                   Impact                    │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Added Stock(Stock other) copy constructor │ Eliminated 12 copy-paste blocks             │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Single-pass stats loop                    │ 5 loops -> 1, in 1 location instead of 3    │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Controllers use StockService exclusively  │ 8 service bypasses removed                  │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ API returns Stock directly                │ 4 stringly-typed Map conversions gone       │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Removed 5 unused methods                  │ getTotalMarketValue, getAverageChange, etc. │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Reusable Random field                     │ No more new Random() per call               │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Streams + enhanced for-each               │ Cleaner iteration                           │
└───────────────────────────────────────────┴─────────────────────────────────────────────┘

The app is running at http://localhost:8082 and the full before/after report is at stock-app/simplify-report.md.
```

## Usage

### After writing a new function
```
> /simplify
```
Claude Code reviews the changed code, finds redundant logic or overly verbose patterns, and rewrites them to be cleaner.

### After a large refactor
```
> /simplify
```
Scans all modified files for duplicated code across functions, consolidates shared logic, and removes unnecessary abstractions.

### Before committing
```
> /simplify
```
Acts as an automated code review pass, catching things like unused variables, inefficient loops, or code that could leverage existing utilities.

### Combined with other work
```
> write a REST API handler for /users
> /simplify
```
After generating code, run `/simplify` to tighten it up, remove any over-engineering, and ensure it follows the simplest possible approach.

## Before and After

### Redundant conditions
Before:
```python
if x is not None:
    if x > 0:
        if x < 100:
            process(x)
```
After:
```python
if x is not None and 0 < x < 100:
    process(x)
```

### Duplicated logic
Before:
```go
func GetUser(id int) (User, error) {
    db := connectDB()
    defer db.Close()
    row := db.QueryRow("SELECT * FROM users WHERE id = ?", id)
    var u User
    err := row.Scan(&u.ID, &u.Name)
    return u, err
}

func GetOrder(id int) (Order, error) {
    db := connectDB()
    defer db.Close()
    row := db.QueryRow("SELECT * FROM orders WHERE id = ?", id)
    var o Order
    err := row.Scan(&o.ID, &o.Total)
    return o, err
}
```
After:
```go
func queryRow[T any](query string, id int, scan func(*sql.Row) (T, error)) (T, error) {
    db := connectDB()
    defer db.Close()
    return scan(db.QueryRow(query, id))
}
```

### Over-engineered code
Before:
```java
public class UserServiceFactory {
    private static UserServiceFactory instance;
    public static UserServiceFactory getInstance() {
        if (instance == null) instance = new UserServiceFactory();
        return instance;
    }
    public UserService create() {
        return new UserService();
    }
}
```
After:
```java
UserService service = new UserService();
```

## /simplify in action

```
❯ /simplify also create a simplify-report.md with code before and after you /simplify changes

⏺ Let me get the diff first, then launch the three review agents.

⏺ Bash(cd /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/claude-code-simplify && git diff HEAD 2>/dev/null | head -20; echo "---"; git
    status --short)
⎿  ---

⏺ Bash(git status)
⎿  On branch main
    Your branch is up to date with 'origin/main'.

    nothing to commit, working tree clean

⏺ No git diff available - all changes were already committed or the files are untracked. Let me check the recently modified files.

⏺ Searched for 1 pattern, read 1 file (ctrl+o to expand)

⏺ Let me read all the source files to pass to the agents.

⏺ Read 4 files (ctrl+o to expand)

⏺ I have all the source code. Now launching the three review agents in parallel.

⏺ Agent(Code reuse review)
⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)

⏺ Agent(Code quality review)
⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)

⏺ Agent(Efficiency review)
⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)

⏺ Three review agents launched in parallel. I'll apply fixes once they complete.

✻ Brewed for 1m 11s · 3 background tasks still running (↓ to manage)

⏺ Agent "Code reuse review" completed

⏺ Agent 1 (Code Reuse) complete. Waiting for the other two agents.

⏺ Agent "Code quality review" completed

⏺ Agent 2 (Code Quality) complete. Waiting for Agent 3 (Efficiency).

⏺ Agent "Efficiency review" completed

⏺ All three agents complete. Now I'll apply all fixes. Key changes:

1. Add copy constructor to Stock
2. Simplify StockService — eliminate copy blocks, single-pass stats, remove redundant methods
3. DashboardController — use StockService, remove StockRepository
4. StockApiController — use StockService, return Stock directly instead of Map
```