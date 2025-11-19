---
name: json-formatter
description: Validate, format, and minify JSON files when users request JSON validation, formatting, or ask to validate their JSONs
allowed-tools: [Read, Bash]
---

# JSON Formatter Skill

Format, validate, and minify JSON files with precision and safety.

## Capabilities

This skill provides comprehensive JSON processing:

- **Validation**: Parse and validate JSON syntax, reporting errors with line numbers
- **Formatting**: Pretty-print JSON with configurable indentation (default 2 spaces)
- **Minification**: Remove all unnecessary whitespace for production use
- **Safety**: Always show diffs before writing changes
- **Metrics**: Report file size before and after operations

## Usage

When a user requests JSON processing, this skill:

1. Reads the target JSON file
2. Validates the syntax
3. Asks user preference: format or minify
4. Shows the proposed changes
5. Reports size difference
6. Writes the result

## Instructions

For any JSON file operation:

- Always validate before processing
- Use 2-space indentation for formatting
- Preserve UTF-8 encoding
- Show clear error messages with line numbers for invalid JSON
- Never modify files without showing diffs first
- Handle nested objects and arrays correctly
- Support both single files and batch operations
