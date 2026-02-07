import { useState, useMemo } from "react";
import { useStore } from "@tanstack/react-store";
import { wizardStore, reset } from "../store/wizard";
import { CodeViewer } from "./CodeViewer";
import styles from "./PreviewStep.module.css";

export function PreviewStep() {
  const generatedHtml = useStore(wizardStore, (s) => s.generatedHtml);
  const [view, setView] = useState<"preview" | "code">("preview");
  const [copied, setCopied] = useState(false);

  const blobUrl = useMemo(() => {
    const blob = new Blob([generatedHtml], { type: "text/html" });
    return URL.createObjectURL(blob);
  }, [generatedHtml]);

  function handleCopy() {
    navigator.clipboard.writeText(generatedHtml);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function handleDownload() {
    const blob = new Blob([generatedHtml], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `solution-${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleStartOver() {
    reset();
  }

  return (
    <div className={styles.container}>
      <div className={styles.toggleBar}>
        <button
          className={`${styles.toggleBtn} ${view === "preview" ? styles.active : ""}`}
          onClick={() => setView("preview")}
        >
          Preview
        </button>
        <button
          className={`${styles.toggleBtn} ${view === "code" ? styles.active : ""}`}
          onClick={() => setView("code")}
        >
          Code
        </button>
      </div>

      {view === "preview" ? (
        <iframe src={blobUrl} className={styles.iframe} title="Preview" />
      ) : (
        <CodeViewer code={generatedHtml} />
      )}

      <div className={styles.actions}>
        <button className={`${styles.actionBtn} ${styles.actionBtnPrimary}`} onClick={handleCopy}>
          Copy HTML
        </button>
        <button className={styles.actionBtn} onClick={handleDownload}>
          Download
        </button>
        <button className={styles.actionBtn} onClick={handleStartOver}>
          Start Over
        </button>
        {copied && <span className={styles.copied}>Copied!</span>}
      </div>
    </div>
  );
}
