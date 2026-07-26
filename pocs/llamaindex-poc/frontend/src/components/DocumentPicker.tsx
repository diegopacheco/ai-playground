import type { DocumentRecord } from "../types";

interface Props {
  documents: DocumentRecord[];
  selected: string[];
  onChange: (docIds: string[]) => void;
  single?: boolean;
}

export function DocumentPicker({ documents, selected, onChange, single }: Props) {
  if (documents.length === 0) {
    return <p className="muted">No documents indexed yet — upload PDFs on the Ingest tab.</p>;
  }

  const toggle = (docId: string): void => {
    if (single) {
      onChange([docId]);
      return;
    }
    onChange(
      selected.includes(docId)
        ? selected.filter((entry) => entry !== docId)
        : [...selected, docId],
    );
  };

  return (
    <div className="row">
      {!single && (
        <button onClick={() => onChange([])} disabled={selected.length === 0}>
          all documents
        </button>
      )}
      {documents.map((document) => (
        <button
          key={document.doc_id}
          className={selected.includes(document.doc_id) ? "primary" : ""}
          onClick={() => toggle(document.doc_id)}
        >
          {document.file_name}
        </button>
      ))}
    </div>
  );
}
