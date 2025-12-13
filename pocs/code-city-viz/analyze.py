#!/usr/bin/env python3
import subprocess
import os
import sys
import json
import shutil

def clone_repo(github_url, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    subprocess.run(['git', 'clone', '--depth', '1', github_url, target_dir], check=True)

def count_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except:
        return 0

def get_file_extension(filepath):
    _, ext = os.path.splitext(filepath)
    return ext.lower() if ext else 'none'

def analyze_codebase(root_dir):
    files_data = []
    skip_dirs = {'.git', 'node_modules', 'vendor', 'target', 'build', 'dist', '__pycache__', '.idea', '.vscode'}
    code_extensions = {
        '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.go', '.rs', '.c', '.cpp', '.h', '.hpp',
        '.cs', '.rb', '.php', '.swift', '.kt', '.scala', '.clj', '.ex', '.exs', '.erl',
        '.hs', '.ml', '.fs', '.r', '.m', '.mm', '.sh', '.bash', '.zsh', '.zig'
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        rel_dir = os.path.relpath(dirpath, root_dir)
        if rel_dir == '.':
            rel_dir = ''

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            ext = get_file_extension(filename)

            if ext not in code_extensions:
                continue

            loc = count_lines(filepath)
            if loc == 0:
                continue

            rel_path = os.path.join(rel_dir, filename) if rel_dir else filename

            files_data.append({
                'path': rel_path,
                'name': filename,
                'extension': ext,
                'loc': loc,
                'directory': rel_dir
            })

    return sorted(files_data, key=lambda x: x['loc'], reverse=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <github_url>")
        sys.exit(1)

    github_url = sys.argv[1]
    target_dir = '/tmp/codecity'

    print(f"Cloning {github_url}...")
    clone_repo(github_url, target_dir)

    print("Analyzing codebase...")
    files_data = analyze_codebase(target_dir)

    output = {
        'repo_url': github_url,
        'total_files': len(files_data),
        'total_loc': sum(f['loc'] for f in files_data),
        'files': files_data
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Analysis complete: {output['total_files']} files, {output['total_loc']} total LOC")
    print(f"Data saved to {output_path}")

if __name__ == '__main__':
    main()
