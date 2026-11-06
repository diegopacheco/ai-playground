# Agentsfile Specification v1

An `Agentsfile` declares the agent setup of a project the way a `Dockerfile` declares an image or a
`main.tf` declares infrastructure. It groups skills, agents, custom commands, MCP servers and
plugins into named profiles, so one file can describe a `dev`, a `review` or a `ci` setup and any
tool can pick the profile it needs.

Every item can be written whole inside the file, or point to an external file with `source`.

The file is named `Agentsfile` and is UTF-8 text. Files with the `.agentsfile` extension follow the
same rules.

## 1. Shape

```hcl
version = "1"

profile "dev" {
  description = "Everyday coding setup"

  skill "code-review" {
    description = "Reviews the current diff"
    content     = <<EOT
      # Code review
      Read the diff and list bugs first.
      EOT

    file "scripts/diff.sh" {
      content = <<SH
        #!/usr/bin/env bash
        git diff --stat
        SH
    }
  }

  agent "tester" {
    model   = "claude-sonnet-5-5"
    tools   = ["Read", "Grep", "Bash"]
    content = <<PROMPT
      You write tests first and keep each test small.
      PROMPT
  }

  command "ship" {
    description   = "Runs tests and opens a PR"
    argument-hint = "<branch>"
    content       = "Run the tests, then open a PR for $ARGUMENTS."
  }

  mcp "github" {
    transport = "stdio"
    command   = "npx"
    args      = ["-y", "@modelcontextprotocol/server-github"]
    env       = { GITHUB_TOKEN = "${GITHUB_TOKEN}" }
  }

  plugin "toolkit" {
    version = "1.2.0"

    skill "lint" {
      content = "Run the linter and fix every error."
    }
  }

  plugin "security" {
    source = "https://github.com/acme/plugins/security"
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
* `${NAME}` inside a string is plain text. Parsers never expand it; the tool that applies the
  profile resolves it from the environment.

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
and their keys must be unique.

## 4. Schema

### File

| Statement | Kind | Required | Notes |
|---|---|---|---|
| `version` | string attribute | yes | must be `"1"` |
| `profile "<name>"` | block | no, repeatable | names unique in the file |

### profile

| Statement | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `skill "<name>"` | block | repeatable, names unique per profile |
| `agent "<name>"` | block | repeatable, names unique per profile |
| `command "<name>"` | block | a custom slash command, names unique per profile |
| `mcp "<name>"` | block | repeatable, names unique per profile |
| `plugin "<name>"` | block | repeatable, names unique per profile |

The same name may be used by two different kinds.

### skill

| Statement | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `source` | string | path or URL of the skill folder |
| `content` | string | the whole `SKILL.md` body, written inline |
| `file "<path>"` | block | an extra file of the skill, paths unique per skill |

A skill sets exactly one of `source` or `content`.

### file

| Attribute | Kind | Notes |
|---|---|---|
| `source` | string | path or URL of the file to copy |
| `content` | string | the whole file, written inline |

A file sets exactly one of `source` or `content`. Its label is the path inside the skill folder.

### agent

| Attribute | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `model` | string | optional |
| `tools` | list | optional, defaults to empty |
| `source` | string | path or URL of the agent definition |
| `content` | string | the whole system prompt, written inline |

An agent sets exactly one of `source` or `content`.

### command

| Attribute | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `argument-hint` | string | optional, shown next to the command, such as `<branch>` |
| `model` | string | optional |
| `tools` | list | optional, the tools the command may use, defaults to empty |
| `source` | string | path or URL of the command prompt |
| `content` | string | the whole command prompt, written inline |

A command sets exactly one of `source` or `content`.

### mcp

| Attribute | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `source` | string | path or URL of a server definition |
| `transport` | string | one of `stdio`, `http`, `sse` |
| `command` | string | required when transport is `stdio` |
| `args` | list | optional, defaults to empty |
| `env` | map | optional, defaults to empty |
| `url` | string | required when transport is `http` or `sse` |
| `headers` | map | optional, defaults to empty |

An MCP server sets exactly one of `source` or `transport`.

### plugin

| Statement | Kind | Notes |
|---|---|---|
| `description` | string | optional |
| `version` | string | optional |
| `author` | string | optional |
| `source` | string | path, git URL or marketplace reference of a published plugin |
| `skill`, `agent`, `command`, `mcp` | blocks | same schemas as in a profile, names unique per kind per plugin |

A plugin sets exactly one of `source` or inline items (at least one nested block).

## 5. Parsing model

Every SDK exposes the same typed model:

```text
Agentsfile { version, profiles[] }
Profile    { name, description?, skills[], agents[], commands[], mcps[], plugins[] }
Skill      { name, description?, source?, content?, files[] }
SkillFile  { path, source?, content? }
Agent      { name, description?, model?, tools[], source?, content? }
Command    { name, description?, argument_hint?, model?, tools[], source?, content? }
Mcp        { name, description?, source?, transport?, command?, args[], env{}, url?, headers{} }
Plugin     { name, description?, version?, author?, source?, skills[], agents[], commands[], mcps[] }
Transport  = stdio | http | sse
```

Profiles, plugins, files and the items of each kind keep the order in which they appear in the file.

## 6. Processing order

A parser stops at the first error and processes the file in three steps:

1. Tokenize the whole file. A lexical error is reported before any other error.
2. Parse the tokens into statements. A syntax error is reported before any schema error.
3. Validate the schema in document order. For each block, its statements are checked in order:
   an attribute is checked for being known, duplicated, for its kind, then for its value
   (`version` and `transport`), and a child block is checked for being known, duplicated, then
   validated recursively. After its statements, the block is checked for what it requires, then
   for conflicts, then for transport rules. The missing `version` is checked last, after every
   profile.

## 7. Errors

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
| schema | the attribute value | `invalid transport "V", expected stdio, http or sse` |
| schema | the block keyword | `duplicate profile "N"` |
| schema | the block keyword | `duplicate KIND "N" in OWNER "P"` |
| schema | the block keyword | `KIND "N" requires "source" or "content"` for skill, file, agent and command |
| schema | the block keyword | `mcp "N" requires "source" or "transport"` |
| schema | the block keyword | `plugin "N" requires "source" or inline items` |
| schema | the block keyword | `KIND "N" cannot set both "source" and "content"` for skill, file, agent and command |
| schema | the block keyword | `mcp "N" cannot set both "source" and "transport"` |
| schema | the block keyword | `plugin "N" cannot set both "source" and inline items` |
| schema | line 1, col 1 | `missing required attribute "version"` |
| schema | the block keyword | `mcp "N" with T transport requires "A"` |

* `T` names the found token: `identifier`, `string`, `'='`, `'{'`, `'}'`, `'['`, `']'`, `','` or
  `end of file`.
* `CTX` is the keyword of the enclosing block (`profile`, `skill`, `file`, `agent`, `command`,
  `mcp`, `plugin`), or `file` at the top level.
* `OWNER` is `profile`, `plugin` or `skill` (for duplicate `file` paths).
* Positions point at the first character of the token named in the Position column.

## 8. Linter

`agentsfile-lint` validates files beyond what a parser must reject. It reports every schema error,
not only the first, and adds warnings:

| Code | Level | Rule |
|---|---|---|
| E001 | error | lexical or syntax error |
| E002 | error | unknown attribute |
| E003 | error | unknown block |
| E004 | error | duplicate attribute |
| E005 | error | wrong attribute kind |
| E006 | error | unsupported version or invalid transport |
| E007 | error | duplicate profile or duplicate item |
| E008 | error | missing `version`, or neither `source` nor the inline definition |
| E009 | error | transport requirement not met |
| E010 | error | both `source` and the inline definition |
| W001 | warning | profile declares nothing |
| W002 | warning | a `source` starting with `.` or `/` does not exist next to the Agentsfile |
| W003 | warning | `env` or `headers` value looks like a hardcoded secret; use `${NAME}` |
| W004 | warning | block name is not kebab-case |
| W005 | warning | attribute ignored by the chosen transport |
| W006 | warning | inline `content` is empty |

Output is one line per diagnostic, `path:line:col: level[CODE]: message`. The exit code is `1`
when there is at least one error, `0` otherwise.

## 9. Conformance

`spec/fixtures/valid/` holds files every SDK must parse. `spec/fixtures/invalid/` holds files every
SDK must reject; the first line of each one is `# expect: <error>` and the SDK error string must be
exactly `<error>`.
