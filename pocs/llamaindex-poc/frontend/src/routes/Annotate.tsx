import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { DocumentPicker } from "../components/DocumentPicker";
import { renderPage } from "../pdf";
import type { AnnotationSaved, Mark } from "../types";

const COLORS = ["#ffe066", "#8ce99a", "#74c0fc", "#ffa8a8", "#e599f7"];
const MIN_SIZE = 0.004;

interface Drag {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function Annotate() {
  const queryClient = useQueryClient();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const originRef = useRef<{ x: number; y: number } | null>(null);
  const dragRef = useRef<Drag | null>(null);

  const [docId, setDocId] = useState("");
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(0);
  const [scale, setScale] = useState(1.3);
  const [tool, setTool] = useState<"highlight" | "note">("highlight");
  const [color, setColor] = useState(COLORS[0]);
  const [marks, setMarks] = useState<Mark[]>([]);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [renderError, setRenderError] = useState("");

  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const saved = useQuery({ queryKey: ["annotations"], queryFn: api.annotations });

  const save = useMutation<AnnotationSaved, Error, void>({
    mutationFn: () => api.saveAnnotations({ doc_id: docId, marks }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["annotations"] }),
  });

  useEffect(() => {
    if (!docId || !canvasRef.current) return;
    setRenderError("");
    renderPage(`/api/documents/${docId}/file`, page, scale, canvasRef.current)
      .then((info) => setPages(info.pages))
      .catch((error: Error) => setRenderError(error.message));
  }, [docId, page, scale]);

  const relative = (event: React.MouseEvent): { x: number; y: number } => {
    const box = stageRef.current?.getBoundingClientRect();
    if (!box) return { x: 0, y: 0 };
    return {
      x: Math.min(Math.max((event.clientX - box.left) / box.width, 0), 1),
      y: Math.min(Math.max((event.clientY - box.top) / box.height, 0), 1),
    };
  };

  const onDown = (event: React.MouseEvent): void => {
    if (!docId) return;
    const point = relative(event);
    if (tool === "note") {
      setMarks((current) => [
        ...current,
        { page, x: point.x, y: point.y, width: 0.03, height: 0.03, color, note: "", kind: "note" },
      ]);
      return;
    }
    originRef.current = point;
    dragRef.current = { x: point.x, y: point.y, width: 0, height: 0 };
    setDrag(dragRef.current);
  };

  const onMove = (event: React.MouseEvent): void => {
    const origin = originRef.current;
    if (!origin) return;
    const point = relative(event);
    dragRef.current = {
      x: Math.min(origin.x, point.x),
      y: Math.min(origin.y, point.y),
      width: Math.abs(point.x - origin.x),
      height: Math.abs(point.y - origin.y),
    };
    setDrag(dragRef.current);
  };

  const onUp = (): void => {
    const current = dragRef.current;
    originRef.current = null;
    dragRef.current = null;
    setDrag(null);
    if (!current || current.width < MIN_SIZE || current.height < MIN_SIZE) return;
    setMarks((existing) => [...existing, { page, ...current, color, note: "", kind: "highlight" }]);
  };

  const setNote = (position: number, note: string): void =>
    setMarks((current) =>
      current.map((mark, index) => (index === position ? { ...mark, note } : mark)),
    );

  const pageMarks = marks.filter((mark) => mark.page === page);

  return (
    <>
      <div className="card">
        <h2>Annotate and highlight a PDF</h2>
        <p className="hint">
          Drag on the page to highlight, or switch to the note tool and click to drop a sticky note.
          Saving writes real <code>/Highlight</code> and <code>/Text</code> annotation objects into a
          new PDF with pypdf — the original file is never modified.
        </p>
        <DocumentPicker
          documents={documents.data?.documents ?? []}
          selected={docId ? [docId] : []}
          onChange={(ids) => {
            setDocId(ids[0] ?? "");
            setMarks([]);
            setPage(1);
          }}
          single
        />
      </div>

      {docId && (
        <div className="card">
          <div className="row" style={{ marginBottom: 12 }}>
            <button className={tool === "highlight" ? "primary" : ""} onClick={() => setTool("highlight")}>
              highlight
            </button>
            <button className={tool === "note" ? "primary" : ""} onClick={() => setTool("note")}>
              note
            </button>
            {COLORS.map((entry) => (
              <button
                key={entry}
                className={`swatch ${color === entry ? "active" : ""}`}
                style={{ background: entry }}
                onClick={() => setColor(entry)}
              />
            ))}
            <span className="grow" />
            <button disabled={page <= 1} onClick={() => setPage((current) => current - 1)}>
              ‹
            </button>
            <span className="muted">
              page {page} / {pages || "…"}
            </span>
            <button disabled={pages > 0 && page >= pages} onClick={() => setPage((current) => current + 1)}>
              ›
            </button>
            <button onClick={() => setScale((current) => Math.max(0.6, current - 0.2))}>−</button>
            <button onClick={() => setScale((current) => Math.min(2.4, current + 0.2))}>+</button>
          </div>

          {renderError && <div className="error">{renderError}</div>}

          <div className="scroll-x">
            <div
              ref={stageRef}
              className="pdf-stage"
              onMouseDown={onDown}
              onMouseMove={onMove}
              onMouseUp={onUp}
              onMouseLeave={onUp}
            >
              <canvas ref={canvasRef} />
              {pageMarks.map((mark, position) => (
                <div
                  key={position}
                  className={`mark ${mark.kind}`}
                  style={{
                    left: `${mark.x * 100}%`,
                    top: `${mark.y * 100}%`,
                    width: `${mark.width * 100}%`,
                    height: `${mark.height * 100}%`,
                    background: mark.kind === "highlight" ? mark.color : undefined,
                  }}
                />
              ))}
              {drag && (
                <div
                  className="mark highlight"
                  style={{
                    left: `${drag.x * 100}%`,
                    top: `${drag.y * 100}%`,
                    width: `${drag.width * 100}%`,
                    height: `${drag.height * 100}%`,
                    background: color,
                  }}
                />
              )}
            </div>
          </div>
        </div>
      )}

      {docId && (
        <div className="card">
          <h2>Annotations ({marks.length})</h2>
          {marks.length === 0 && <p className="muted">nothing marked yet</p>}
          {marks.map((mark, position) => (
            <div className="row" key={position} style={{ marginBottom: 8 }}>
              <span className="badge">p{mark.page}</span>
              <span className="swatch" style={{ background: mark.color, cursor: "default" }} />
              <span className="muted" style={{ width: 68 }}>
                {mark.kind}
              </span>
              <input
                type="text"
                className="grow"
                value={mark.note}
                placeholder="note text (optional for highlights)"
                onChange={(event) => setNote(position, event.target.value)}
              />
              <button
                className="danger"
                onClick={() => setMarks((current) => current.filter((_, index) => index !== position))}
              >
                remove
              </button>
            </div>
          ))}
          <div className="row" style={{ marginTop: 12 }}>
            <button className="primary" disabled={marks.length === 0 || save.isPending} onClick={() => save.mutate()}>
              {save.isPending ? "writing…" : "Save as new PDF"}
            </button>
            <button disabled={marks.length === 0} onClick={() => setMarks([])}>
              clear
            </button>
          </div>
          {save.isError && (
            <div className="error" style={{ marginTop: 10 }}>
              {save.error.message}
            </div>
          )}
          {save.data && (
            <p style={{ marginBottom: 0 }}>
              Wrote {save.data.applied} annotations —{" "}
              <a href={save.data.url} target="_blank" rel="noreferrer">
                {save.data.name}
              </a>
            </p>
          )}
        </div>
      )}

      <div className="card">
        <h2>Saved annotated PDFs</h2>
        {(saved.data?.files.length ?? 0) === 0 && <p className="muted">none yet</p>}
        <ul style={{ marginBottom: 0 }}>
          {saved.data?.files.map((file) => (
            <li key={file.name}>
              <a href={`/api/annotations/${file.name}`} target="_blank" rel="noreferrer">
                {file.name}
              </a>{" "}
              <span className="muted">({(file.size_bytes / 1024).toFixed(1)} KB)</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}
