---
name: dead-code
description: Find dead Java code by static reachability from entry points. Parses the sources, builds the method call graph, walks it from real entry points (main, Spring web/lifecycle/scheduler handlers, JUnit tests, framework callbacks), and renders a light-theme website with summary cards, a reachable-vs-dead chart, and a grouped list of every unreachable method and dead class. Use when the user runs /dead-code or asks to find dead, unused, or unreachable code, or which methods and classes are safe to delete.
---

# dead-code

Generate a static dead-code report website for the Java project the user is working on.

## Steps

1. Find the Java source root. Prefer `src/main/java`; if absent, use the directory the user points at or the closest directory containing `.java` files. If the user names a specific module or subproject, scope to it. To include tests, point the script at a root that also covers `src/test/java`.
2. Run the analyzer that ships with this skill:

   ```
   python3 ~/.claude/skills/dead-code/find_dead_code.py <java-src-dir> <out-dir> [title]
   ```

   Use `dead-code-site` inside the target project as `<out-dir>` unless the user asks for another location, and pass the project name as the title.
3. The script writes a single self-contained `index.html` and prints the totals (methods, entry points, reachable, dead, dead classes). Serve it with `python3 -m http.server` from `<out-dir>` (pick a free port) or tell the user to open the file directly, then report the URL or path and the headline numbers.
4. Tell the user how to read the site: the cards summarize totals, the donut shows the reachable-vs-dead split, the **dead code** tab lists every unreachable method grouped by file (with `file:line`, a reason tag, and who calls it), and the **entry points** tab shows the seeds the analysis started from. The search box filters by class, method, or file.

## How reachability is decided

- Entry points (reachability roots) are: `main(...)`; methods annotated with Spring web mappings (`@GetMapping`, `@PostMapping`, `@RequestMapping`, ...), `@Bean`, `@PostConstruct`, `@PreDestroy`, `@EventListener`, `@Scheduled`, messaging listeners, `@ExceptionHandler`; JUnit `@Test` and lifecycle methods; and public framework callbacks (`run`, `call`, `doFilter`, ...) on classes implementing `CommandLineRunner`, `Runnable`, `Filter`, and similar.
- A method is reachable if it is an entry point or is called, directly or transitively, from one. Everything else is reported as dead.
- Dead methods are tagged `no references` (nobody calls them) or `only dead callers` (called only from other dead code). A class whose every method is dead is flagged as a dead class.
- Calls are resolved from source text by the declared type of fields, parameters and locals, plus `implements`/`extends` relationships so a call through an interface reaches its implementations.

## Notes

- The site is light themed and hand-drawn in style: wobbly SVG, pastel fills, handwriting fonts.
- The analyzer needs only the Python standard library; do not install anything.
- Static analysis cannot see reflection, dynamic proxies, dependency-injection by interface that is wired at runtime, serialization, or methods invoked only from configuration or external frameworks. Treat the report as a strong candidate list and review each item before deleting.
- If the script reports no methods found, confirm the path actually contains `.java` files before retrying.
