import React, { useEffect, useMemo, useState } from 'react';
import { fetchTree, fetchSource } from '../api.js';
import { highlightLine } from '../lib/highlight.js';

export default function CodeViewer({ report, query }) {
  const [tree, setTree] = useState([]);
  const [selected, setSelected] = useState(null);
  const [source, setSource] = useState('');

  useEffect(() => {
    fetchTree().then(setTree).catch(() => setTree([]));
  }, []);

  useEffect(() => {
    if (!selected) return;
    fetchSource(selected).then(setSource).catch(() => setSource('could not load file'));
  }, [selected]);

  const q = query.trim().toLowerCase();
  const files = q ? tree.filter(f => f.path.toLowerCase().includes(q)) : tree;

  const ccByLine = useMemo(() => {
    const map = {};
    if (selected) {
      for (const fn of report.complexity.functions) {
        if (fn.file === selected) map[fn.line] = fn.cyclomatic;
      }
    }
    return map;
  }, [selected, report]);

  const lines = source ? source.split('\n') : [];
  const threshold = report.complexity.ccThreshold;

  return (
    <div className="code">
      <aside className="filetree">
        {files.map(f => (
          <button
            key={f.path}
            className={selected === f.path ? 'file active' : 'file'}
            onClick={() => setSelected(f.path)}
          >
            <span className="file-path">{f.path}</span>
            <span className="file-loc">{f.loc}</span>
          </button>
        ))}
        {files.length === 0 && <div className="muted pad">no matching files</div>}
      </aside>
      <section className="codepane">
        {!selected && <div className="muted pad">select a file to view its source</div>}
        {selected && (
          <>
            <div className="codepane-head">{selected}</div>
            <div className="codelines">
              {lines.map((line, i) => {
                const cc = ccByLine[i + 1];
                return (
                  <div className="codeline" key={i}>
                    <span className="ln">{i + 1}</span>
                    <span className="cc">
                      {cc != null && (
                        <em className={cc > threshold ? 'cc-bad' : 'cc-ok'}>CC {cc}</em>
                      )}
                    </span>
                    <code className="src" dangerouslySetInnerHTML={{ __html: highlightLine(line) || '&nbsp;' }} />
                  </div>
                );
              })}
            </div>
          </>
        )}
      </section>
    </div>
  );
}
