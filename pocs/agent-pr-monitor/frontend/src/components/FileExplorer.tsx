import { useState } from "react";
import type { FileEntry } from "../types";

interface Props {
  files: FileEntry[];
  selectedPath: string | null;
  onSelectFile: (path: string) => void;
}

function TreeNode({
  entry,
  depth,
  selectedPath,
  onSelectFile,
}: {
  entry: FileEntry;
  depth: number;
  selectedPath: string | null;
  onSelectFile: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(depth < 1);

  if (entry.is_dir) {
    return (
      <div>
        <div
          className="file-tree-item"
          style={{ paddingLeft: depth * 16 }}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "[v]" : "[>]"} {entry.name}/
        </div>
        {expanded &&
          entry.children.map((child) => (
            <TreeNode
              key={child.path}
              entry={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelectFile={onSelectFile}
            />
          ))}
      </div>
    );
  }

  return (
    <div
      className={`file-tree-item ${selectedPath === entry.path ? "selected" : ""}`}
      style={{ paddingLeft: depth * 16 }}
      onClick={() => onSelectFile(entry.path)}
    >
      {"  "} {entry.name}
    </div>
  );
}

export default function FileExplorer({
  files,
  selectedPath,
  onSelectFile,
}: Props) {
  return (
    <div className="file-explorer">
      <h3>Files</h3>
      <div className="file-tree">
        {files.map((entry) => (
          <TreeNode
            key={entry.path}
            entry={entry}
            depth={0}
            selectedPath={selectedPath}
            onSelectFile={onSelectFile}
          />
        ))}
      </div>
    </div>
  );
}
