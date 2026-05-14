# repo-mcp-poc

`repo-mcp` is a local MCP server that lets Claude Code work with GitHub repositories as first-class context. You register a repo once (it gets cloned to `~/.mcp/repo-tool/repos/`) and Claude can then call tools like `list_repos`, `tree`, `read_file`, `grep`, `list_files`, `git_log`, and `repo_info` against it — so questions like "what modules does Spring have?" or "how does ConcurrentLruCache work?" are answered straight from the source code instead of guesses or stale training data.

See [design-doc.md](./design-doc.md) for architecture and design details.

# Usaage Sample

```
❯ ./install.sh
repo-mcp already registered in /Users/diegopacheco/.claude.json
Add a GitHub repo now? [y/N] y
GitHub URL: https://github.com/spring-projects/spring-framework
Branch (optional, blank = default):
{
  "name": "spring-projects__spring-framework",
  "path": "/Users/diegopacheco/.mcp/repo-tool/repos/spring-projects__spring-framework"
}
Add another GitHub repo? [y/N] n

Installed to /Users/diegopacheco/.mcp/repo-tool
Entry:      /Users/diegopacheco/.mcp/repo-tool/bin/repo-mcp
Registry:   /Users/diegopacheco/.mcp/repo-tool/registry.json
Re-run install.sh to add more repos, or call add_repo from Claude Code.
Uninstall:  /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/repo-mcp-poc/uninstall.sh
❯ cat /Users/diegopacheco/.mcp/repo-tool/registry.json
{
  "version": 1,
  "repos": [
    {
      "name": "aidenybai__million",
      "url": "https://github.com/aidenybai/million",
      "path": "/Users/diegopacheco/.mcp/repo-tool/repos/aidenybai__million",
      "added_at": "2026-05-14T04:32:27.997Z"
    },
    {
      "name": "spring-projects__spring-framework",
      "url": "https://github.com/spring-projects/spring-framework",
      "path": "/Users/diegopacheco/.mcp/repo-tool/repos/spring-projects__spring-framework",
      "added_at": "2026-05-14T04:39:45.831Z"
    }
  ]
}%                                                                                                                                                    ❯ claude
 ▐▛███▜▌   Claude Code v2.1.140
▝▜█████▛▘  Opus 4.7 with high effort · Claude Pro
  ▘▘ ▝▝    ~/git/diegopacheco/ai-playground/pocs/repo-mcp-poc

❯ using repo_mcp what modules spring has?

  Called repo-mcp 2 times (ctrl+o to expand)

⏺ Spring Framework has the following modules (per repo-mcp tree of spring-projects/spring-framework):

  Core
  - spring-core
  - spring-core-test
  - spring-beans
  - spring-context
  - spring-context-indexer
  - spring-context-support
  - spring-expression

  AOP & Instrumentation
  - spring-aop
  - spring-aspects
  - spring-instrument

  Data Access
  - spring-jdbc
  - spring-r2dbc
  - spring-orm
  - spring-oxm
  - spring-tx

  Messaging
  - spring-jms
  - spring-messaging

  Web
  - spring-web
  - spring-webmvc
  - spring-webflux
  - spring-websocket

  Testing
  - spring-test

  Build/Aggregator
  - framework-api
  - framework-bom
  - framework-platform
  - framework-docs
  - integration-tests
  - buildSrc

✻ Crunched for 13s

❯ using repo_mcp what features spring-core has?

  Called repo-mcp (ctrl+o to expand)

⏺ spring-core features (per package contents):

  AOT (Ahead-Of-Time) — aot/
  - Native image hints, code generation, GraalVM support (AotDetector, generate/, hint/, nativex/)

  Bytecode
  - Bundled ASM (asm/) — class reading/writing, annotation/method visitors
  - Bundled CGLIB (cglib/) — proxying, reflection, transforms
  - Bundled Objenesis (objenesis/) — instantiation without constructors
  - Bundled JavaPoet (javapoet/)

  Core abstractions — core/
  - Type resolution: ResolvableType, GenericTypeResolver, ParameterizedTypeReference, MethodParameter
  - Ordering: Ordered, PriorityOrdered, OrderComparator
  - Exceptions: NestedRuntimeException, NestedCheckedException
  - ClassLoaders: DecoratingClassLoader, OverridingClassLoader, SmartClassLoader
  - Kotlin/Native detection: KotlinDetector, NativeDetector
  - Parameter name discovery (ParameterNameDiscoverer + Kotlin/Standard variants)
  - Reactive adapters: ReactiveAdapterRegistry, ReactiveAdapter
  - Coroutines: CoroutinesUtils, PropagationContextElement
  - Thread-locals: NamedThreadLocal, NamedInheritableThreadLocal
  - Sub-packages: annotation/, codec/, convert/, env/, io/ (Resource abstraction), log/, metrics/, retry/, serializer/, style/, task/, type/

  Nullability annotations — lang/
  - @Nullable, @NonNull, @NonNullApi, @NonNullFields, @Contract, @CheckReturnValue

  Utilities — util/
  - Assertions (Assert), reflection (ReflectionUtils, ClassUtils)
  - Collections: LinkedMultiValueMap, ConcurrentReferenceHashMap, ConcurrentLruCache, LinkedCaseInsensitiveMap, CompositeMap/Set/Collection
  - Strings: StringUtils, PatternMatchUtils, PlaceholderParser, PropertyPlaceholderHelper
  - Path matching: AntPathMatcher, PathMatcher, RouteMatcher
  - IO: FileCopyUtils, FileSystemUtils, StreamUtils, FastByteArrayOutputStream
  - MIME types: MimeType, MimeTypeUtils
  - Misc: StopWatch, DigestUtils, NumberUtils, ObjectUtils, SerializationUtils, IdGenerator
  - Sub-packages: backoff/, comparator/, concurrent/, function/, unit/, xml/

  Want me to drill into any sub-package (e.g. core/io, core/convert, core/env)?

✻ Sautéed for 15s

❯ using repo_mcp how ConcurrentLruCache works?

  Called repo-mcp (ctrl+o to expand)

⏺ Here's how ConcurrentLruCache works (file: spring-core/src/main/java/org/springframework/util/ConcurrentLruCache.java).

  High-level idea

  It's a bounded LRU cache inspired by Ben Manes' ConcurrentLinkedHashMap. The key trick: reads and writes are recorded into buffers and drained later
   under a single lock, instead of mutating the LRU order on every access. This avoids contention on hot paths.

  Core state

  - ConcurrentMap<K, Node<K,V>> cache — actual storage (ConcurrentHashMap) (line 57)
  - EvictionQueue<K,V> evictionQueue — a doubly-linked list ordered LRU-first (line 71, 526)
  - ReadOperations — per-thread ring buffers for "this node was read" events (line 368)
  - WriteOperations — a ConcurrentLinkedQueue<Runnable> of add/remove tasks (line 461)
  - ReentrantLock evictionLock — only one thread mutates the LRU list at a time (line 65)
  - AtomicReference<DrainStatus> — IDLE | REQUIRED | PROCESSING to coordinate drains (line 73, 313)

  get(key) flow (line 101)

  1. If capacity == 0 → bypass, always generate.
  2. Look up in ConcurrentHashMap.
  3. Miss: call generator.apply(key), then put(...) which builds a Node, putIfAbsent into the map, and enqueues an AddTask write op.
  4. Hit: call processRead(node) — records the read into a thread-local buffer, then maybe drains.

  Read path — striped buffers (line 368)

  - Number of buffers = min(4, nextPowerOfTwo(cpus)) (line 453).
  - Each thread picks a buffer by Thread.getId() & mask (line 405) — reduces contention.
  - recordRead does lazySet into an AtomicReferenceArray slot — fast, no CAS (line 409).
  - Returns true if pending ops < 32 (MAX_PENDING_OPERATIONS), meaning the drain can be delayed.

  Write path (line 137)

  - Enqueues task, sets drainStatus = REQUIRED, attempts a drain.
  - Writes always force a drain; reads only force one when the buffer fills.

  Drain — the only place LRU order changes (line 143)

  - Uses tryLock (non-blocking). If another thread is draining, the current thread skips it.
  - Holder of the lock:
    a. Drains read buffers → each recorded node is moveToBack in the eviction queue (most-recently-used goes to tail) (line 437–451).
    b. Drains up to 16 write tasks (DRAIN_THRESHOLD, line 463) — runs AddTask/RemovalTask.
  - AddTask.run() adds node to the queue tail and calls evictEntries(), which polls from the head (LRU) and removes from the map until currentSize <=
  capacity (line 278–287).

  Node lifecycle (line 354)

  States ACTIVE → PENDING_REMOVAL → REMOVED via CAS on the Node's AtomicReference<CacheEntry>. remove(key) first marks as pending so an in-flight
  AddTask won't add a doomed node to the queue.

  Why this design

  - Reads = striped lazy writes to a ring buffer — near lock-free.
  - LRU ordering and eviction are batched under a single lock — no contention storm.
  - ConcurrentHashMap answers hits without ever touching the eviction queue.

  Net result: get/put are O(1) amortized with very low contention, while still respecting strict LRU eviction.

✻ Cooked for 23s
```