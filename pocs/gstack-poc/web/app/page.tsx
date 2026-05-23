export default function Home() {
  return (
    <main
      style={{
        padding: 32,
        maxWidth: 720,
        margin: "0 auto",
        lineHeight: 1.6,
      }}
    >
      <h1 style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 28 }}>
        qa2pw
      </h1>
      <p style={{ color: "#71717A", marginTop: 0 }}>
        Plain English in. Real Playwright out.
      </p>

      <h2>Status</h2>
      <p>
        The three-pane playground UI lands in T7 (see <code>DESIGN.md</code> and{" "}
        <code>preview.html</code> for the visual). Until then, this page is just
        a heartbeat to confirm the Next.js scaffold is up.
      </p>

      <h2>API smoke test</h2>
      <p>
        The <code>POST /api/generate</code> endpoint is wired and streams SSE.
        From a second terminal:
      </p>
      <pre
        style={{
          background: "#F4F4F5",
          padding: 16,
          borderRadius: 4,
          overflow: "auto",
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 13,
        }}
      >
        {`curl -N -X POST http://127.0.0.1:3000/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt":"log in with standard_user/secret_sauce, see Products","url":"https://www.saucedemo.com"}'`}
      </pre>
      <p>
        Make sure Ollama is up and <code>qwen2.5vl:32b</code> (or set{" "}
        <code>QA2PW_MODEL=qwen2.5vl:7b</code>) is pulled before sending. The
        endpoint walks Playwright + Claude-style vision tool calls and returns a
        streamed <code>result</code> event with the generated{" "}
        <code>.spec.ts</code>.
      </p>

      <h2>What is built so far</h2>
      <ul>
        <li>
          <strong>runner</strong> package — Ollama vision loop, step counter
          (25) + 90s wall clock, deterministic <code>.spec.ts</code> templater,
          allowlist with safety blocklist, Playwright session with guaranteed
          dispose-on-error. 63 tests.
        </li>
        <li>
          <strong>web</strong> package — Next.js 15 App Router, this page, and
          the <code>/api/generate</code> SSE endpoint above.
        </li>
      </ul>
    </main>
  );
}
