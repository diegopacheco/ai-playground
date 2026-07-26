import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import { DocumentPicker } from "../components/DocumentPicker";
import type { RustResponse } from "../types";

export function Rust() {
  const [docIds, setDocIds] = useState<string[]>([]);
  const [question, setQuestion] = useState("Summarise this document in five bullet points.");

  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const status = useQuery({ queryKey: ["rust-status"], queryFn: api.rustStatus });

  const ask = useMutation<RustResponse, Error, void>({
    mutationFn: () => api.rustAsk({ doc_id: docIds[0], question }),
  });

  const ready = status.data?.binary_ready && status.data?.model_ready;

  return (
    <>
      <div className="card">
        <h2>Rust llama.cpp over a single PDF</h2>
        <p className="hint">
          The backend shells out to a Rust binary that extracts the PDF text with{" "}
          <code>pdf-extract</code> and runs it through a local GGUF model with{" "}
          <code>llama-cpp-2</code>. No Python, no Ollama, no vector index in this path.
        </p>
        <div className="row">
          <span className={`badge ${status.data?.binary_ready ? "on" : "off"}`}>
            binary {status.data?.binary_ready ? "ready" : "missing"}
          </span>
          <span className={`badge ${status.data?.model_ready ? "on" : "off"}`}>
            gguf {status.data?.model_ready ? "ready" : "missing"}
          </span>
        </div>
        <p className="muted" style={{ fontSize: 12, marginBottom: 0 }}>
          <code>{status.data?.binary}</code>
          <br />
          <code>{status.data?.model}</code>
        </p>
        {!ready && status.data && (
          <div className="error" style={{ marginTop: 10 }}>
            Run <code>./build.sh</code> to compile the Rust binary and make sure the GGUF path exists.
          </div>
        )}
      </div>

      <div className="card">
        <h2>Pick one PDF</h2>
        <DocumentPicker
          documents={documents.data?.documents ?? []}
          selected={docIds}
          onChange={setDocIds}
          single
        />
        <textarea
          style={{ marginTop: 10 }}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <div className="row" style={{ marginTop: 10 }}>
          <button
            className="primary"
            disabled={docIds.length === 0 || !question.trim() || ask.isPending || !ready}
            onClick={() => ask.mutate()}
          >
            {ask.isPending ? "loading model and generating…" : "Run Rust llama"}
          </button>
          <span className="muted">first call loads the whole GGUF, expect a slow start</span>
        </div>
        {ask.isError && (
          <div className="error" style={{ marginTop: 10 }}>
            {ask.error.message}
          </div>
        )}
      </div>

      {ask.data && (
        <div className="card">
          <h2>{ask.data.file_name}</h2>
          <p className="hint">
            {ask.data.chars_extracted.toLocaleString()} chars extracted ·{" "}
            {ask.data.chars_sent.toLocaleString()} sent to the model
            {ask.data.truncated ? " (truncated)" : ""} · {ask.data.elapsed_seconds}s
          </p>
          <div className="bubble">{ask.data.answer}</div>
        </div>
      )}
    </>
  );
}
