# /simplify

`/simplify` is a bundled Claude Code slash command that reviews changed code for reuse, quality, and efficiency, then automatically fixes any issues found.

## What it does

- Reviews code for **reuse opportunities** (duplicated logic, patterns that could be consolidated)
- Assesses **code quality** (readability, maintainability, best practices)
- Checks **efficiency** (unnecessary complexity, performance improvements)
- **Fixes issues** automatically by applying simplifications directly to the code

## How to use

Type `/simplify` in Claude Code after making code changes. It analyzes your current changes and applies improvements.

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
