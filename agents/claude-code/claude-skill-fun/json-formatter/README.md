# JSON Formatter Skill

A Claude skill for formatting, validating, and minifying JSON files.

## Structure

```
json-formatter/
├── SKILL.md          - Skill instructions for Claude
├── validate.py       - JSON validation script
├── format.py         - JSON formatting/minifying script
└── README.md         - This file
```

## Scripts

### validate.py
Validates JSON syntax and reports errors.

```bash
./validate.py <filepath>
```

### format.py
Formats or minifies JSON files.

```bash
./format.py <filepath>          # Format with 2-space indentation
./format.py <filepath> --minify # Minify (remove whitespace)
```

## Features

- Syntax validation with detailed error reporting
- Pretty-print formatting with 2-space indentation
- Minification for production use
- File size comparison
- UTF-8 encoding preservation
- No external dependencies (uses Python standard library)
