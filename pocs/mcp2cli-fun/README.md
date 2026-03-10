# MCP2CLI

https://github.com/knowsuchagency/mcp2cli

## Install

```
uvx mcp2cli --help
```

## List tools from mcp server

```
❯ uvx mcp2cli --mcp-stdio "npx @modelcontextprotocol/server-filesystem /tmp" --list

Available tools:
  read-file                                 Read the complete contents of a file as text. DEPRECATED: Use read_tex
  read-text-file                            Read the complete contents of a file from the file system as text. Han
  read-media-file                           Read an image or audio file. Returns the base64 encoded data and MIME
  read-multiple-files                       Read the contents of multiple files simultaneously. This is more effic
  write-file                                Create a new file or completely overwrite an existing file with new co
  edit-file                                 Make line-based edits to a text file. Each edit replaces exact line se
  create-directory                          Create a new directory or ensure a directory exists. Can create multip
  list-directory                            Get a detailed listing of all files and directories in a specified pat
  list-directory-with-sizes                 Get a detailed listing of all files and directories in a specified pat
  directory-tree                            Get a recursive tree view of files and directories as a JSON structure
  move-file                                 Move or rename files and directories. Can move files between directori
  search-files                              Recursively search for files and directories matching a pattern. The p
  get-file-info                             Retrieve detailed metadata about a file or directory. Returns comprehe
  list-allowed-directories                  Returns the list of directories that this server is allowed to access.
```

