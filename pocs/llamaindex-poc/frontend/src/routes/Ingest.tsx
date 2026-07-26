import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { useRef, useState } from "react";
import { api } from "../api";
import type { DocumentRecord, UploadResponse } from "../types";

const columnHelper = createColumnHelper<DocumentRecord>();

function kb(bytes: number): string {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

export function Ingest() {
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [sorting, setSorting] = useState<SortingState>([{ id: "ingested_at", desc: true }]);

  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });

  const invalidate = (): void => {
    void queryClient.invalidateQueries({ queryKey: ["documents"] });
    void queryClient.invalidateQueries({ queryKey: ["health"] });
  };

  const upload = useMutation<UploadResponse, Error, File[]>({
    mutationFn: api.upload,
    onSuccess: invalidate,
  });

  const remove = useMutation<unknown, Error, string>({
    mutationFn: api.deleteDocument,
    onSuccess: invalidate,
  });

  const accept = (list: FileList | null): void => {
    if (!list) return;
    const files = Array.from(list).filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    if (files.length > 0) upload.mutate(files);
  };

  const columns = [
    columnHelper.accessor("file_name", { header: "File" }),
    columnHelper.accessor("pages", { header: "Pages" }),
    columnHelper.accessor("chunks", { header: "Chunks" }),
    columnHelper.accessor("chars", { header: "Chars" }),
    columnHelper.accessor("size_bytes", {
      header: "Size",
      cell: (info) => kb(info.getValue()),
    }),
    columnHelper.accessor("ingested_at", {
      header: "Indexed",
      cell: (info) => info.getValue().replace("T", " ").replace("+00:00", " UTC"),
    }),
    columnHelper.display({
      id: "actions",
      header: "",
      cell: (info) => (
        <button
          className="danger"
          disabled={remove.isPending}
          onClick={() => remove.mutate(info.row.original.doc_id)}
        >
          delete
        </button>
      ),
    }),
  ];

  const table = useReactTable({
    data: documents.data?.documents ?? [],
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const stats = documents.data?.stats;

  return (
    <>
      <div className="card">
        <h2>Drop PDFs to parse and index</h2>
        <p className="hint">
          Each PDF is parsed page by page with the LlamaIndex PDFReader, split by SentenceSplitter,
          embedded with Ollama and persisted to a vector index on disk.
        </p>
        <div
          className={`dropzone ${over ? "over" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            setOver(false);
            accept(event.dataTransfer.files);
          }}
        >
          {upload.isPending ? "parsing and embedding…" : "Drop PDF files here, or click to choose"}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          multiple
          hidden
          onChange={(event) => {
            accept(event.target.files);
            event.target.value = "";
          }}
        />
        {upload.isError && (
          <div className="error" style={{ marginTop: 12 }}>
            {upload.error.message}
          </div>
        )}
        {upload.data && (
          <ul style={{ marginBottom: 0 }}>
            {upload.data.results.map((result, position) => (
              <li key={position}>
                <b>{result.file_name}</b> — {result.status}
                {result.detail ? `: ${result.detail}` : ""}
                {result.status === "indexed" ? ` (${result.pages}p, ${result.chunks} chunks)` : ""}
              </li>
            ))}
          </ul>
        )}
      </div>

      {stats && (
        <div className="card">
          <h2>Index</h2>
          <div className="stats">
            <div className="stat">
              <b>{stats.documents}</b>
              <span>documents</span>
            </div>
            <div className="stat">
              <b>{stats.pages}</b>
              <span>pages</span>
            </div>
            <div className="stat">
              <b>{stats.chunks}</b>
              <span>chunks</span>
            </div>
            <div className="stat">
              <b>{stats.chars.toLocaleString()}</b>
              <span>characters</span>
            </div>
            <div className="stat">
              <b>{stats.chunk_size}</b>
              <span>chunk size</span>
            </div>
            <div className="stat">
              <b>{stats.chunk_overlap}</b>
              <span>overlap</span>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <h2>Indexed documents</h2>
        <p className="hint">Click a column header to sort. Deleting drops its nodes from the index.</p>
        {remove.isError && <div className="error">{remove.error.message}</div>}
        <div className="scroll-x">
          <table>
            <thead>
              {table.getHeaderGroups().map((group) => (
                <tr key={group.id}>
                  {group.headers.map((header) => (
                    <th key={header.id} onClick={header.column.getToggleSortingHandler()}>
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {{ asc: " ▲", desc: " ▼" }[header.column.getIsSorted() as string] ?? ""}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.length === 0 && (
                <tr>
                  <td colSpan={columns.length} className="muted">
                    nothing indexed yet
                  </td>
                </tr>
              )}
              {table.getRowModel().rows.map((row) => (
                <tr key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
