# Agentsfile Specification v1

An `Agentsfile` declares the agent setup of a project the way a `Dockerfile` declares an image or a
`main.tf` declares infrastructure. It groups skills, agents, custom commands and MCP servers into
named profiles, so one file can describe a `dev`, a `review` or a `ci` setup and any tool can pick
the profile it needs.

The file is named `Agentsfile` and is UTF-8 text. Files with the `.agentsfile` extension follow the
same rules.

## 1. Shape

```hcl
version = "1"

profile "dev" {
  description = "Everyday coding setup"

  skill "code-review" {
    source      = "./skills/code-review"
    description = "Reviews the current diff"
  }

  agent "tester" {
    source = "./agents/tester.md"
    model  = "claude-sonnet-5-5"
    tools  = ["Read", "Grep", "Bash"]
  }

  command "ship" {
    source = "./commands/ship.md"
  }

  mcp "github" {
    transport = "stdio"
    command   = "npx"
    args      = ["-y", "@modelcontextprotocol/server-github"]
    env       = { GITHUB_TOKEN = "${GITHUB_TOKEN}" }
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
| punctuation | `=` `{` `}` `[` `]` `,` |

* Lines and columns start at 1. Columns count characters, not bytes.
* A string that reaches a newline or the end of the file is an `unterminated string`.
* `${NAME}` inside a string is plain text. Parsers never expand it; the tool that applies the
  profile resolves it from the environment.

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

Lists hold strings only. Maps hold string values only and their keys must be unique.

## 4. Schema

### File

| Statement | Kind | Required | Notes |
|---|---|---|---|
| `version` | string attribute | yes | must be `"1"` |
| `profile "<name>"` | block | no, repeatable | names unique in the file |

### profile

| Statement | Kind | Required | Notes |
|---|---|---|---|
| `description` | string | no | |
| `skill "<name>"` | block | repeatable | names unique per profile |
| `agent "<name>"` | block | repeatable | names unique per profile |
| `command "<name>"` | block | repeatable | a custom slash command, names unique per profile |
| `mcp "<name>"` | block | repeatable | names unique per profile |

### skill

| Attribute | Kind | Required |
|---|---|---|
| `source` | string | yes, path or URL of the skill folder |
| `description` | string | no |

### agent

| Attribute | Kind | Required |
|---|---|---|
| `source` | string | yes, path or URL of the agent definition |
| `description` | string | no |
| `model` | string | no |
| `tools` | list | no, defaults to empty |

### command

| Attribute | Kind | Required |
|---|---|---|
| `source` | string | yes, path or URL of the command prompt |
| `description` | string | no |

### mcp

| Attribute | Kind | Required |
|---|---|---|
| `transport` | string | yes, one of `stdio`, `http`, `sse` |
| `description` | string | no |
| `command` | string | yes when transport is `stdio` |
| `args` | list | no, defaults to empty |
| `env` | map | no, defaults to empty |
| `url` | string | yes when transport is `http` or `sse` |
| `headers` | map | no, defaults to empty |

Skill, agent, command and MCP blocks hold attributes only, never nested blocks.

## 5. Parsing model

Every SDK exposes the same typed model:

```text
Agentsfile { version, profiles[] }
Profile    { name, description?, skills[], agents[], commands[], mcps[] }
Skill      { name, source, description? }
Agent      { name, source, description?, model?, tools[] }
Command    { name, source, description? }
Mcp        { name, transport, description?, command?, args[], env{}, url?, headers{} }
Transport  = stdio | http | sse
```

Profiles and the items of each kind keep the order in which they appear in the file.

## 6. Processing order

A parser stops at the first error and processes the file in three steps:

1. Tokenize the whole file. A lexical error is reported before any other error.
2. Parse the tokens into statements. A syntax error is reported before any schema error.
3. Validate the schema in document order. For each block, its statements are checked in order:
   an attribute is checked for being known, duplicated, for its kind, then for its value
   (`version` and `transport`), and a child block is
   checked for being known, duplicated, then validated recursively. After its statements, the block
   is checked for missing required attributes and then for transport rules. The missing `version`
   is checked last, after every profile.

## 7. Errors

Every SDK reports an error as `line L, col C: message`, with these exact messages:

| Step | Position | Message |
|---|---|---|
| lex | the character | `unexpected character 'X'` |
| lex | the opening quote | `unterminated string` |
| lex | the backslash | `invalid escape '\X'` |
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
| schema | the block keyword | `duplicate KIND "N" in profile "P"` |
| schema | the block keyword | `missing required attribute "A" in KIND "N"` |
| schema | line 1, col 1 | `missing required attribute "version"` |
| schema | the block keyword | `mcp "N" with T transport requires "A"` |

* `T` names the found token: `identifier`, `string`, `'='`, `'{'`, `'}'`, `'['`, `']'`, `','` or
  `end of file`.
* `CTX` is `file`, `profile`, `skill`, `agent`, `command` or `mcp`.
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
| E007 | error | duplicate profile or duplicate item in a profile |
| E008 | error | missing required attribute |
| E009 | error | transport requirement not met |
| W001 | warning | profile declares nothing |
| W002 | warning | local `source` path does not exist next to the Agentsfile |
| W003 | warning | `env` or `headers` value looks like a hardcoded secret; use `${NAME}` |
| W004 | warning | block name is not kebab-case |
| W005 | warning | attribute ignored by the chosen transport |

Output is one line per diagnostic, `path:line:col: level[CODE]: message`. The exit code is `1`
when there is at least one error, `0` otherwise.

## 9. Conformance

`spec/fixtures/valid/` holds files every SDK must parse. `spec/fixtures/invalid/` holds files every
SDK must reject; the first line of each one is `# expect: <error>` and the SDK error string must be
exactly `<error>`.
