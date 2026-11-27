import { useForm } from "@tanstack/react-form";
import { useNavigate } from "@tanstack/react-router";
import { acceptsFile, formatBytes } from "../format";
import { useFormats, useUpload } from "../hooks";

type UploadValues = { file: File | null; ocr: boolean };

const defaultValues: UploadValues = { file: null, ocr: false };

export function UploadPanel() {
  const formats = useFormats();
  const upload = useUpload();
  const navigate = useNavigate();
  const extensions = formats.data ?? [];

  const form = useForm({
    defaultValues,
    onSubmit: async ({ value }) => {
      if (!value.file) return;
      const conversion = await upload.mutateAsync({ file: value.file, ocr: value.ocr });
      await navigate({ to: "/conversions/$id", params: { id: conversion.id } });
    },
  });

  return (
    <form
      className="card upload"
      onSubmit={(e) => {
        e.preventDefault();
        void form.handleSubmit();
      }}
    >
      <h2>Convert a document</h2>
      <p className="muted">Docling parses layout, reading order, tables and headings, then exports Markdown, HTML and RAG chunks.</p>
      <form.Field
        name="file"
        validators={{
          onChange: ({ value }) =>
            value && !acceptsFile(value.name, extensions) ? "This file type is not supported" : undefined,
        }}
      >
        {(field) => (
          <label className="drop">
            <input
              type="file"
              accept={extensions.join(",")}
              onChange={(e) => field.handleChange(e.target.files?.[0] ?? null)}
            />
            <span className="drop-title">{field.state.value ? field.state.value.name : "Choose a file"}</span>
            <span className="muted">
              {field.state.value ? formatBytes(field.state.value.size) : extensions.join("  ")}
            </span>
            {field.state.meta.errors.length > 0 && <span className="error">{field.state.meta.errors.join(", ")}</span>}
          </label>
        )}
      </form.Field>
      <form.Field name="ocr">
        {(field) => (
          <label className="check">
            <input type="checkbox" checked={field.state.value} onChange={(e) => field.handleChange(e.target.checked)} />
            Run OCR (for scanned PDFs and images, slower)
          </label>
        )}
      </form.Field>
      <form.Subscribe selector={(s) => [s.values.file, s.canSubmit] as const}>
        {([file, canSubmit]) => (
          <button type="submit" disabled={!file || !canSubmit || upload.isPending}>
            {upload.isPending ? "Converting..." : "Convert"}
          </button>
        )}
      </form.Subscribe>
      {upload.isError && <p className="error">{upload.error.message}</p>}
    </form>
  );
}
