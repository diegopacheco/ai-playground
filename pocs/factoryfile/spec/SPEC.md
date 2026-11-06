# Factoryfile Specification v1.1

A `Factoryfile` declares a software factory for coding agents the way a `Dockerfile` declares an
image or a `main.tf` declares infrastructure. A factory is a graph: its nodes are stages, its
forward edges are the `next` lists and the `on` routes of the stages, and its back edges are named
loops with an exit condition and a maximum number of iterations. A stage is worked by one or more
agents, or runs a command of the runtime such as the test suite.

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
and their keys must be unique. The only map attribute is `verdict`.

## 4. Schema

### File

| Statement | Kind | Required | Notes |
|---|---|---|---|
| `version` | string attribute | yes | must be `"1"` |
| `factory "<name>"` | block | no, repeatable | names unique in the file |

### factory

| Statement | Kind | Notes |
|---|---|---|
| `title` | string | optional, the display name, such as `"Approval Gate"`; defaults to the block name |
| `description` | string | optional |
| `entry` | string | optional, the stage where work starts; defaults to the first stage |
| `flow` | string | optional, how work enters the factory, such as `"push"` or `"pull"` |
| `backlog` | string | optional, `"required"` or `"optional"`, whether the factory needs a backlog item to start |
| `agent "<name>"` | block | repeatable, names unique per factory |
| `stage "<name>"` | block | repeatable, names unique per factory |
| `loop "<name>"` | block | repeatable, names unique per factory |

The same name may be used by two different kinds.

### agent

| Attribute | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `role` | string | optional, such as `"product owner"` |
| `provider` | string | optional, the runtime that runs the agent, such as `"claude"` or `"codex"` |
| `model` | string | optional |
| `tools` | list | optional, defaults to empty |
| `source` | string | path or URL of the agent definition |
| `content` | string | the whole system prompt, written inline |

An agent sets exactly one of `source` or `content`.

### stage

| Statement | Kind | Notes |
|---|---|---|
| `label` | string | optional, the display name of the stage, such as `"Approved?"` |
| `description` | string | optional |
| `agents` | list | names of agents of the same factory; required unless `run` is set |
| `run` | string | optional, a command the runtime runs instead of an agent, such as `"test"`, `"ship"`, `"merge"`, `"finish"` or `"script"` |
| `script` | string | optional, the path of the script a `run = "script"` stage runs |
| `next` | list | optional, names of the stages that follow; a stage with no `next` and no `on` route ends the factory |
| `artifacts` | list | optional, what the stage produces, such as `"design.md"` |
| `output` | string | optional, the one file the stage must write, such as `"approval.json"` |
| `check` | string | optional, how the runtime checks the stage, such as `"output"`, `"verdict"` or `"harness"` |
| `verdict` | map | optional, how to read a decision from `output`, see below |
| `source` | string | optional, path or URL of the instructions of the stage |
| `content` | string | optional, the instructions of the stage, written inline |
| `skills` | list | optional, paths of skill files given to the agents of the stage |
| `tasks` | string | optional, a stage whose `output` lists tasks; the stage runs once per task, in parallel |
| `parallel` | string | optional, the most tasks run at once, `[1-9][0-9]{0,5}` |
| `visits` | string | optional, how many times the stage may run in one cycle, `[1-9][0-9]{0,5}` |
| `end` | string | optional, the status the factory ends with when this stage runs, such as `"succeeded"` |
| `at` | list | optional, the column and row of the stage in a drawing, two integers from `0` to `999` |
| `on "<outcome>"` | block | repeatable, where work goes for one outcome of the stage, see below |

A stage sets at most one of `source` or `content`.

`verdict` keys are `field` (required), `yes`, `no`, `list` and `notes`, all strings: `field` names
the boolean field of the `output` JSON, `yes` and `no` name the outcomes taken when it is true or
false, `list` and `notes` name fields that explain the decision.

### on

An `on` route is a forward or back edge taken for one outcome of its stage, such as `"passed"` and
`"failed"` for a test run, or the `yes` and `no` outcomes of a verdict. The route named `"next"` is
the edge every outcome takes, the long form of a one item `next` list. A stage uses either the next
route (`next` or `on "next"`) or outcome routes, not both.

| Statement | Kind | Notes |
|---|---|---|
| `to` | string | required, a stage of the same factory |
| `label` | string | optional, the text drawn on the edge |
| `reason` | string | optional, why work takes this route, kept with the work |
| `cycle` | string | optional, `"true"` when the route starts a new cycle of the factory, defaults to `"false"` |
| `exhausted "<stage>"` | block | optional, once, where work goes instead when `to` has run as often as allowed |

```hcl
stage "test" {
  label = "Test"
  run   = "test"
  at    = ["2", "0"]

  on "passed" {
    to    = "approve"
    label = "pass"
  }

  on "failed" {
    to    = "implement"
    label = "fail"

    exhausted "postmortem" {
      label  = "attempts exhausted"
      reason = "tests kept failing after every implementation attempt"
    }
  }
}

stage "approve" {
  agents  = ["approver"]
  source  = "prompts/approve.md"
  output  = "approval.json"
  check   = "verdict"
  verdict = { field = "approved", yes = "approved", no = "rejected", list = "reasons" }

  on "approved" { to = "ship" }
  on "rejected" {
    to     = "postmortem"
    reason = "the approver denied the change"
  }
}
```

### exhausted

`exhausted "<stage>"` names the stage that takes the work when the target of the enclosing route,
or of the enclosing `exhausted`, already ran as often as allowed.

| Statement | Kind | Notes |
|---|---|---|
| `label` | string | optional, the text drawn on the edge |
| `reason` | string | optional, why work takes this route |
| `exhausted "<stage>"` | block | optional, once, the next fallback when this one is exhausted too |

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

`entry`, the items of `agents` and `next`, `tasks`, the `to` of an `on` route, the label of an
`exhausted` block, and `from` and `to` name other blocks of the same factory. A name is known when a block of the right kind with that label appears anywhere in the
factory, before or after the reference.

References are checked after every statement of the factory has been validated. They are checked
in the order the referencing attributes appear in the file, and the items of a list in their
order. Only attributes that were accepted (known, not duplicated and of the right kind) are
checked.

## 6. Parsing model

Every SDK exposes the same typed model:

```text
Factoryfile { version, factories[] }
Factory     { name, title?, description?, entry?, flow?, backlog?, agents[], stages[], loops[] }
Agent       { name, description?, role?, provider?, model?, tools[], source?, content? }
Stage       { name, label?, description?, agents[], run?, script?, next[], artifacts[], output?,
              check?, verdict?, source?, content?, skills[], tasks?, parallel?, visits?, end?,
              at[], routes[] }
Route       { outcome, to, label?, reason?, cycle, exhausted? }
Exhausted   { to, label?, reason?, exhausted? }
Loop        { name, description?, from, to, until?, max? }
```

* `entry` is the `entry` attribute when it is set, otherwise the name of the first stage, and it
  is absent when the factory has no stage.
* `max`, `parallel`, `visits` and the items of `at` are integers, `cycle` is a boolean.
* `verdict` is a map from key to string.
* `routes` holds the `on` blocks of the stage in file order, `outcome` being the block label.
* Factories and the items of each kind keep the order in which they appear in the file.

## 7. Processing order

A parser stops at the first error and processes the file in three steps:

1. Tokenize the whole file. A lexical error is reported before any other error.
2. Parse the tokens into statements. A syntax error is reported before any schema error.
3. Validate the schema in document order. For each block, its statements are checked in order:
   an attribute is checked for being known, duplicated, for its kind, then for its value
   (`version`, `max`, `parallel`, `visits`, `at`, `cycle` and `verdict`), and a child block is checked for being known, duplicated, then
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
| schema | the attribute name | `attribute "A" must be a string`, `a list` or `a map` |
| schema | the attribute value | `unsupported version "V"` |
| schema | the attribute value | `invalid max "V", expected an integer from 1 to 999999`, the same for `parallel` and `visits` |
| schema | the attribute value | `invalid at, expected two integers from 0 to 999` |
| schema | the attribute value | `invalid cycle "V", expected "true" or "false"` |
| schema | the attribute value | `unknown key "K" in verdict`, keys checked in byte order |
| schema | the attribute value | `verdict requires "field"` |
| schema | the block keyword | `duplicate factory "N"` |
| schema | the block keyword | `duplicate KIND "N" in factory "F"` |
| schema | the block keyword | `duplicate on "O" in stage "S"` |
| schema | the block keyword | `duplicate exhausted in CTX "N"` |
| schema | the block keyword | `agent "N" requires "source" or "content"` |
| schema | the block keyword | `agent "N" cannot set both "source" and "content"` |
| schema | the block keyword | `stage "N" requires "agents" or "run"` |
| schema | the block keyword | `stage "N" cannot set both "source" and "content"` |
| schema | the block keyword | `stage "N" cannot set both "next" and on "next"` |
| schema | the block keyword | `stage "N" cannot mix the next route with on "O"` |
| schema | the block keyword | `on "O" requires "to"` |
| schema | the block keyword | `loop "N" requires "from"`, then `loop "N" requires "to"` |
| schema | the attribute value | `unknown stage "X" in factory "F"` for `entry` |
| schema | the attribute value | `unknown agent "X" in stage "S"` for `agents` |
| schema | the attribute value | `unknown stage "X" in stage "S"` for `next`, `tasks`, the `to` of its routes and the label of their `exhausted` blocks |
| schema | the attribute value | `unknown stage "X" in loop "L"` for `from` and `to` |
| schema | line 1, col 1 | `missing required attribute "version"` |

* `T` names the found token: `identifier`, `string`, `'='`, `'{'`, `'}'`, `'['`, `']'`, `','` or
  `end of file`.
* `CTX` is the keyword of the enclosing block (`factory`, `agent`, `stage`, `on`, `exhausted`,
  `loop`), or `file` at the top level. In `duplicate exhausted in CTX "N"`, `CTX "N"` is the
  enclosing `on` or `exhausted` block.
* `KIND` is `agent`, `stage` or `loop`.
* A wrong kind reports `a string`, `a list` or `a map`.
* A stage reports `requires` first, then `source` with `content`, then `next` with `on "next"`,
  then the next route mixed with the first outcome route.
* The label of an `exhausted` block is a reference reported at the position of its `exhausted` keyword.
* Positions point at the first character of the token named in the Position column. For a list
  that is its `[`.
* When `loop` misses both `from` and `to`, a parser reports `from`, the first one.

## 9. Linter

`factoryfile-lint` validates files beyond what a parser must reject. It reports every schema
error, not only the first, and adds warnings:

| Code | Level | Rule |
|---|---|---|
| E001 | error | lexical or syntax error |
| E002 | error | unknown attribute or unknown `verdict` key |
| E003 | error | unknown block |
| E004 | error | duplicate attribute |
| E005 | error | wrong attribute kind |
| E006 | error | unsupported version, or invalid `max`, `parallel`, `visits`, `at` or `cycle` |
| E007 | error | duplicate factory, item, `on` route or `exhausted` |
| E008 | error | missing `version`, `agents` or `run`, `from`, `to`, the `to` of a route, the `field` of a verdict, or neither `source` nor `content` |
| E009 | error | unknown stage or agent reference |
| E010 | error | both `source` and `content`, or both the next route and outcome routes |
| W001 | warning | the factory declares no stage |
| W002 | warning | an agent `source` starting with `.` or `/` does not exist next to the Factoryfile |
| W003 | warning | a stage cannot be reached from the entry over `next`, routes, `exhausted` and loops |
| W004 | warning | a block name is not kebab-case |
| W005 | warning | an agent is not used by any stage |
| W006 | warning | an inline `content` is empty |
| W007 | warning | `next` or a route goes back to an earlier stage, a cycle that no loop, `cycle` or `exhausted` bounds |
| W008 | warning | a loop has no `max`, so it may never end |

Output is one line per diagnostic, `path:line:col: level[CODE]: message`. The exit code is `1`
when there is at least one error, `0` otherwise.

## 10. Visualizer

`factory-cli` draws the graph of each factory:

* `factory-cli --render [path] [--factory NAME]` prints the stages layer by layer in the terminal,
  with their agents or command, their `next` edges, their routes with outcome and label, the
  `exhausted` fallbacks and their loops.
* `factory-cli --render-ui [path] [--factory NAME] [--out FILE] [--no-open]` writes one
  self-contained HTML page with an SVG graph and opens it in the browser.

The path defaults to `./Factoryfile`. When every stage sets `at`, stages are drawn at their column
and row. Otherwise they are placed in layers by their longest distance from the entry over `next`
edges and routes, ignoring edges that close a cycle. Loops are drawn as back edges with their
`until` condition and `max`, `exhausted` fallbacks as dashed edges, and command stages with their
`run` in place of agents.

## 11. Conformance

`spec/fixtures/valid/` holds files every SDK must parse. `spec/fixtures/invalid/` holds files every
SDK must reject; the first line of each one is `# expect: <error>` and the SDK error string must be
exactly `<error>`.

## 12. Changes in v1.1

v1.1 only adds optional statements, so every v1 file is a v1.1 file and `version` stays `"1"`.

* factory: `title`, `flow`, `backlog`.
* agent: `provider`.
* stage: `label`, `run`, `script`, `output`, `check`, `verdict`, `source`, `skills`, `tasks`,
  `parallel`, `visits`, `end`, `at` and `on` routes with `exhausted` fallbacks. A stage that sets
  `run` needs no `agents`.
* The error `stage "N" requires "agents"` became `stage "N" requires "agents" or "run"`.

With these, a Factoryfile holds everything a runtime such as Softorio needs to run the factory, so a
runtime manifest like `factory.yaml` can be generated from the Factoryfile with nothing lost.
