#!/usr/bin/env python3
import json
import sys

def validate_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
        print(f"Valid JSON: {filepath}")
        return 0
    except json.JSONDecodeError as e:
        print(f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"File not found: {filepath}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate.py <filepath>", file=sys.stderr)
        sys.exit(1)

    sys.exit(validate_json(sys.argv[1]))
