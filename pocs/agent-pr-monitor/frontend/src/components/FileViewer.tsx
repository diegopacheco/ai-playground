import { useEffect, useRef } from "react";
import hljs from "highlight.js/lib/core";
import rust from "highlight.js/lib/languages/rust";
import go from "highlight.js/lib/languages/go";
import typescript from "highlight.js/lib/languages/typescript";
import java from "highlight.js/lib/languages/java";
import json from "highlight.js/lib/languages/json";
import yaml from "highlight.js/lib/languages/yaml";
import xml from "highlight.js/lib/languages/xml";
import markdown from "highlight.js/lib/languages/markdown";
import ini from "highlight.js/lib/languages/ini";
import "highlight.js/styles/github-dark.css";

hljs.registerLanguage("rust", rust);
hljs.registerLanguage("go", go);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("java", java);
hljs.registerLanguage("json", json);
hljs.registerLanguage("yaml", yaml);
hljs.registerLanguage("xml", xml);
hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("toml", ini);

interface Props {
  path: string | null;
  content: string | null;
  loading: boolean;
}

function detectLanguage(path: string): string | undefined {
  const ext = path.split(".").pop()?.toLowerCase();
  const map: Record<string, string> = {
    rs: "rust",
    go: "go",
    ts: "typescript",
    tsx: "typescript",
    js: "typescript",
    jsx: "typescript",
    java: "java",
    json: "json",
    yml: "yaml",
    yaml: "yaml",
    toml: "toml",
    xml: "xml",
    html: "xml",
    md: "markdown",
  };
  return ext ? map[ext] : undefined;
}

export default function FileViewer({ path, content, loading }: Props) {
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (codeRef.current && content !== null && path) {
      const lang = detectLanguage(path);
      if (lang) {
        try {
          const result = hljs.highlight(content, { language: lang });
          codeRef.current.innerHTML = result.value;
        } catch {
          codeRef.current.textContent = content;
        }
      } else {
        codeRef.current.textContent = content;
      }
    }
  }, [content, path]);

  if (!path) {
    return (
      <div className="file-viewer">
        <div className="file-viewer-placeholder">Select a file to view</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="file-viewer">
        <div className="file-viewer-header">{path}</div>
        <div className="file-viewer-placeholder">Loading...</div>
      </div>
    );
  }

  const lines = content?.split("\n") ?? [];

  return (
    <div className="file-viewer">
      <div className="file-viewer-header">{path}</div>
      <div className="file-viewer-body">
        <div className="line-numbers">
          {lines.map((_, i) => (
            <div key={i}>{i + 1}</div>
          ))}
        </div>
        <pre className="code-content">
          <code ref={codeRef}>{content}</code>
        </pre>
      </div>
    </div>
  );
}
