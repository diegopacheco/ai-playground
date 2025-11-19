#!/usr/bin/env python3
import json
import sys
import os

def format_json(filepath, minify=False):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        original_size = os.path.getsize(filepath)

        if minify:
            output = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        else:
            output = json.dumps(data, ensure_ascii=False, indent=2)

        print(output)

        temp_size = len(output.encode('utf-8'))
        size_diff = temp_size - original_size
        sign = '+' if size_diff > 0 else ''

        print(f"\n{'='*50}", file=sys.stderr)
        print(f"Original size: {original_size} bytes", file=sys.stderr)
        print(f"New size: {temp_size} bytes ({sign}{size_diff} bytes)", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)

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
        print("Usage: format.py <filepath> [--minify]", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]
    minify = "--minify" in sys.argv

    sys.exit(format_json(filepath, minify))
