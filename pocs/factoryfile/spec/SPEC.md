# Factoryfile Specification v1

A `Factoryfile` declares a software factory for coding agents the way a `Dockerfile` declares an
image or a `main.tf` declares infrastructure. A factory is a graph: its nodes are stages, its
forward edges are the `next` lists of the stages, and its back edges are named loops with an exit
condition and a maximum number of iterations. Each stage is worked by one or more agents.

One file can hold several factories, such as `scrum`, `xp` or `rup`, and a tool picks the one it
needs.

The file is named `Factoryfile` and is UTF-8 text. Files with the `.factoryfile` extension follow
the same rules.

## 1. Shape

```hcl
version = "1"

factory "xp" {
  description = "Extreme Programming with pairs of agents"
  entry       = "planning-game"

  agent "customer" {
    role    = "on-site customer"
    model   = "claude-opus-5-5"
    tools   = ["Read"]
    content = "You write user stories and accept them."
  }

  agent "pair" {
    role    = "driver and navigator"
    tools   = ["Read", "Edit", "Bash"]
    content = <<PROMPT
      You write the test first, then the code that makes it pass.
      PROMPT
  }

  stage "planning-game" {
    agents    = ["customer", "pair"]
    next      = ["test-first"]
    artifacts = ["stories.md"]
  }

  stage "test-first" {
    agents = ["pair"]
    next   = ["acceptance"]
  }

  stage "acceptance" {
    agents  = ["customer"]
    content = "Run the acceptance tests of every story of the iteration."
  }

  loop "tdd" {
    from  = "test-first"
    to    = "test-first"
    until = "every test is green and the code is clean"
    max   = "20"
  }

  loop "iteration" {
    from  = "acceptance"
    to    = "planning-game"
    until = "the customer has no stories left"
    max   = "12"
  }
}
```

## 2. Lexical rules

| Token | Rule |
|---|---|
| whitespace | spaces, tabs, `\r` and `\n` are insignificant and separate tokens |
| comment | `#` up to the end of the line, ignored |
| identifier | `[A-Za-z_][A-Za-z0-9_-]*` |
| string | `"` ... `"` on one line, escapes `\"` `\\` `\n` `\t` |
| heredoc | `<<MARKER` up to a line holding only `MARKER`, see below; it is a string token |
| punctuation | `=` `{` `}` `[` `]` `,` |

* Lines and columns start at 1. Columns count characters, not bytes.
* A string that reaches a newline or the end of the file is an `unterminated string`.
* `${NAME}` inside a string is plain text. Parsers never expand it.

### Heredoc

```hcl
content = <<EOT
  first line
    indented line
  EOT
```

* `MARKER` is `[A-Za-z_][A-Za-z0-9_]*`, written right after `<<`. Only spaces, tabs or `\r` may
  follow it on that line.
* The body is every line after the marker line up to the first line whose text, with spaces, tabs
  and `\r` trimmed from both ends, equals `MARKER`. That closing line is consumed whole.
* A line is blank when it holds only spaces, tabs or `\r`. Blank lines become empty lines.
* The common indentation is the smallest count of leading spaces and tabs over the non-blank lines.
  That many characters are removed from the start of every non-blank line, and a trailing `\r` is
  removed from every line.
* The value is the lines joined with `\n`, with no trailing newline added. A body with no lines is
  the empty string.
* Nothing inside a heredoc is special: no escapes and no comments.

## 3. Grammar

```ebnf
file      = { statement } ;
statement = attribute | block ;
attribute = identifier "=" value ;
block     = identifier string "{" { statement } "}" ;
value     = string | list | map ;
list      = "[" [ string { "," string } [ "," ] ] "]" ;
map       = "{" { identifier "=" string [ "," ] } "}" ;
```

`string` means a quoted string or a heredoc. Lists hold strings only. Maps hold string values only
and their keys must be unique. The v1 schema uses no map attribute, so a map is always the wrong
kind, but it still parses.

## 4. Schema

### File

| Statement | Kind | Required | Notes |
|---|---|---|---|
| `version` | string attribute | yes | must be `"1"` |
| `factory "<name>"` | block | no, repeatable | names unique in the file |

### factory

| Statement | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `entry` | string | optional, the stage where work starts; defaults to the first stage |
| `agent "<name>"` | block | repeatable, names unique per factory |
| `stage "<name>"` | block | repeatable, names unique per factory |
| `loop "<name>"` | block | repeatable, names unique per factory |

The same name may be used by two different kinds.

### agent

| Attribute | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `role` | string | optional, such as `"product owner"` |
| `model` | string | optional |
| `tools` | list | optional, defaults to empty |
| `source` | string | path or URL of the agent definition |
| `content` | string | the whole system prompt, written inline |

An agent sets exactly one of `source` or `content`.

### stage

| Attribute | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `agents` | list | required, names of agents of the same factory |
| `next` | list | optional, names of the stages that follow; empty means the stage ends the factory |
| `artifacts` | list | optional, what the stage produces, such as `"design.md"` |
| `content` | string | optional, the instructions of the stage |

### loop

A loop is a back edge: when stage `from` finishes and `until` is not met yet, work goes back to
stage `to`. `from` and `to` may be the same stage.

| Attribute | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `from` | string | required, a stage of the same factory |
| `to` | string | required, a stage of the same factory |
| `until` | string | optional, the exit condition |
| `max` | string | optional, the maximum number of iterations, `[1-9][0-9]{0,5}` |

## 5. References

`entry`, the items of `agents` and `next`, and `from` and `to` name other blocks of the same
factory. A name is known when a block of the right kind with that label appears anywhere in the
factory, before or after the reference.

References are checked after every statement of the factory has been validated. They are checked
in the order the referencing attributes appear in the file, and the items of a list in their
order. Only attributes that were accepted (known, not duplicated and of the right kind) are
checked.

## 6. Parsing model

Every SDK exposes the same typed model:

```text
Factoryfile { version, factories[] }
Factory     { name, description?, entry?, agents[], stages[], loops[] }
Agent       { name, description?, role?, model?, tools[], source?, content? }
Stage       { name, description?, agents[], next[], artifacts[], content? }
Loop        { name, description?, from, to, until?, max? }
```

* `entry` is the `entry` attribute when it is set, otherwise the name of the first stage, and it
  is absent when the factory has no stage.
* `max` is an integer.
* Factories and the items of each kind keep the order in which they appear in the file.

## 7. Processing order

A parser stops at the first error and processes the file in three steps:

1. Tokenize the whole file. A lexical error is reported before any other error.
2. Parse the tokens into statements. A syntax error is reported before any schema error.
3. Validate the schema in document order. For each block, its statements are checked in order:
   an attribute is checked for being known, duplicated, for its kind, then for its value
   (`version` and `max`), and a child block is checked for being known, duplicated, then
   validated recursively. After its statements, the block is checked for what it requires, then
   for conflicts. A factory then checks its references (section 5). The missing `version` is
   checked last, after every factory.

## 8. Errors

Every SDK reports an error as `line L, col C: message`, with these exact messages:

| Step | Position | Message |
|---|---|---|
| lex | the character | `unexpected character 'X'` |
| lex | the opening quote | `unterminated string` |
| lex | the backslash | `invalid escape '\X'` |
| lex | the `<<` | `invalid heredoc marker` |
| lex | the `<<` | `unterminated heredoc` |
| parse | the token | `expected identifier, found T` at the start of a top-level statement |
| parse | the token | `expected identifier or '}', found T` at the start of a statement inside a block |
| parse | the token | `expected '=' or block label, found T` after a statement identifier |
| parse | the token | `expected '{', found T` after a block label |
| parse | the token | `expected value, found T` after `=` |
| parse | the token | `expected string, found T` for a list item or a map value |
| parse | the token | `expected ',' or ']', found T` after a list item |
| parse | the token | `expected identifier or '}', found T` for a map key |
| parse | the token | `expected '=', found T` after a map key |
| parse | the key | `duplicate key "K"` |
| schema | the attribute name | `unknown attribute "A" in CTX` |
| schema | the block keyword | `unknown block "B" in CTX` |
| schema | the attribute name | `duplicate attribute "A" in CTX` |
| schema | the attribute name | `attribute "A" must be a string` or `a list` |
| schema | the attribute value | `unsupported version "V"` |
| schema | the attribute value | `invalid max "V", expected an integer from 1 to 999999` |
| schema | the block keyword | `duplicate factory "N"` |
| schema | the block keyword | `duplicate KIND "N" in factory "F"` |
| schema | the block keyword | `agent "N" requires "source" or "content"` |
| schema | the block keyword | `agent "N" cannot set both "source" and "content"` |
| schema | the block keyword | `stage "N" requires "agents"` |
| schema | the block keyword | `loop "N" requires "from"`, then `loop "N" requires "to"` |
| schema | the attribute value | `unknown stage "X" in factory "F"` for `entry` |
| schema | the attribute value | `unknown agent "X" in stage "S"` for `agents` |
| schema | the attribute value | `unknown stage "X" in stage "S"` for `next` |
| schema | the attribute value | `unknown stage "X" in loop "L"` for `from` and `to` |
| schema | line 1, col 1 | `missing required attribute "version"` |

* `T` names the found token: `identifier`, `string`, `'='`, `'{'`, `'}'`, `'['`, `']'`, `','` or
  `end of file`.
* `CTX` is the keyword of the enclosing block (`factory`, `agent`, `stage`, `loop`), or `file` at
  the top level.
* `KIND` is `agent`, `stage` or `loop`.
* A wrong kind reports `a string` for string attributes and `a list` for list attributes.
* Positions point at the first character of the token named in the Position column. For a list
  that is its `[`.
* When `loop` misses both `from` and `to`, a parser reports `from`, the first one.

## 9. Linter

`factoryfile-lint` validates files beyond what a parser must reject. It reports every schema
error, not only the first, and adds warnings:

| Code | Level | Rule |
|---|---|---|
| E001 | error | lexical or syntax error |
| E002 | error | unknown attribute |
| E003 | error | unknown block |
| E004 | error | duplicate attribute |
| E005 | error | wrong attribute kind |
| E006 | error | unsupported version or invalid `max` |
| E007 | error | duplicate factory or duplicate item |
| E008 | error | missing `version`, `agents`, `from`, `to`, or neither `source` nor `content` |
| E009 | error | unknown stage or agent reference |
| E010 | error | both `source` and `content` |
| W001 | warning | the factory declares no stage |
| W002 | warning | a `source` starting with `.` or `/` does not exist next to the Factoryfile |
| W003 | warning | a stage cannot be reached from the entry |
| W004 | warning | a block name is not kebab-case |
| W005 | warning | an agent is not used by any stage |
| W006 | warning | an inline `content` is empty |
| W007 | warning | `next` goes back to an earlier stage, a cycle that no loop bounds |
| W008 | warning | a loop has no `max`, so it may never end |

Output is one line per diagnostic, `path:line:col: level[CODE]: message`. The exit code is `1`
when there is at least one error, `0` otherwise.

## 10. Visualizer

`factory-cli` draws the graph of each factory:

* `factory-cli --render [path] [--factory NAME]` prints the stages layer by layer in the terminal,
  with their agents, their `next` edges and their loops.
* `factory-cli --render-ui [path] [--factory NAME] [--out FILE] [--no-open]` writes one
  self-contained HTML page with an SVG graph and opens it in the browser.

The path defaults to `./Factoryfile`. Stages are placed in layers by their longest distance from
the entry over `next` edges, ignoring edges that close a cycle. Loops are drawn as back edges with
their `until` condition and `max`.

## 11. Conformance

`spec/fixtures/valid/` holds files every SDK must parse. `spec/fixtures/invalid/` holds files every
SDK must reject; the first line of each one is `# expect: <error>` and the SDK error string must be
exactly `<error>`.
