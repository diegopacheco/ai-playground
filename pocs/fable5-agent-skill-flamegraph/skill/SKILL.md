---
name: flamegraph
description: Build a function call-graph flamegraph website for a Java codebase. Parses the sources, extracts every method and the calls between them, and renders a light-theme site where the user picks any class or method and sees its call tree as a hand-drawn SVG flamegraph. Use when the user runs /flamegraph or asks for a call graph, call-graph flamegraph, or a visualization of which methods call which.
---

# flamegraph

Generate a static call-graph flamegraph website for the Java project the user is working on.

## Steps

1. Find the Java source root. Prefer `src/main/java`; if absent, use the directory the user points at or the closest directory containing `.java` files. If the user names a specific module or subproject, scope to it.
2. Run the generator that ships with this skill:

   ```
   python3 ~/.claude/skills/flamegraph/build_flamegraph.py <java-src-dir> <out-dir> [title]
   ```

   Use `flamegraph-site` inside the target project as `<out-dir>` unless the user asks for another location, and pass the project name as the title.
3. The script writes a single self-contained `index.html`. Serve it with `python3 -m http.server` from `<out-dir>` (pick a free port) or tell the user to open the file directly, then report the URL or path.
4. Tell the user how to use the site: the left panel lists every class and method, the search box filters it, clicking any method roots the flamegraph at that method, clicking a frame re-roots into it, and the breadcrumb trail walks back up. Frame width is proportional to the number of distinct call paths underneath, hovering shows file, line, and fan-in/fan-out, and a `↺` mark means a recursive cycle was pruned.

## Notes

- The site is light themed and hand-drawn in style: wobbly SVG frames, pastel fills per class, handwriting fonts.
- The call graph is static: it is resolved from source text via field, parameter, and local-variable types, so dynamic dispatch through interfaces resolves only when the declared type itself defines the method.
- The generator needs only the Python standard library; do not install anything.
- If the script reports no methods found, confirm the path actually contains `.java` files before retrying.
