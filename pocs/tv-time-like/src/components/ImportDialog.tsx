import { useRef, useState } from "react"
import type { ImportReport } from "../../shared/types"
import { Icon } from "./Icon"

export function ImportDialog({ open, onClose, onImport, busy }: { open: boolean; onClose: () => void; onImport: (text: string) => Promise<ImportReport>; busy: boolean }) {
  const input = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [report, setReport] = useState<ImportReport | null>(null)
  if (!open) return null
  const run = async () => {
    if (!file) return
    setReport(await onImport(await file.text()))
  }
  return <div className="dialog-backdrop" role="presentation" onMouseDown={event => event.target === event.currentTarget && onClose()}>
    <section className="import-dialog" role="dialog" aria-modal="true" aria-labelledby="import-title">
      <button className="dialog-close" onClick={onClose} aria-label="Close"><Icon name="close"/></button>
      <span className="eyebrow">Bring your history home</span>
      <h2 id="import-title">Import from TV Time</h2>
      <p>Upload <strong>tracking-prod-records-v2.csv</strong> from your official GDPR archive. Matching happens against TVmaze and your local catalog.</p>
      <button className={file ? "drop-zone selected" : "drop-zone"} onClick={() => input.current?.click()}>
        <Icon name={file ? "check" : "upload"} size={26}/>
        <strong>{file ? file.name : "Choose your CSV file"}</strong>
        <span>{file ? `${Math.round(file.size / 1024)} KB ready` : "The file stays on your machine and local server"}</span>
      </button>
      <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={event => { setFile(event.target.files?.[0] || null); setReport(null) }}/>
      {report && <div className="import-report"><strong>{report.imported} watched episodes imported</strong><span>{report.skipped} rows skipped</span>{report.messages.map(message => <small key={message}>{message}</small>)}</div>}
      <div className="dialog-actions"><button className="button secondary" onClick={onClose}>Cancel</button><button className="button primary" onClick={run} disabled={!file || busy}>{busy ? "Importing…" : "Import history"}</button></div>
      <small className="privacy-note">Need the archive? Request it at gdpr.tvtime.com before July 15, 2026.</small>
    </section>
  </div>
}
