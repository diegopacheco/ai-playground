import { useState } from "react";
import { Badge } from "../Badge/Badge";
import "./Tree.css";

export interface TreeNode {
  name: string;
  kind: string;
  detail?: string | null;
  children?: TreeNode[];
}

export interface TreeProps {
  nodes: TreeNode[];
  onSelect?: (node: TreeNode, path: string[]) => void;
  emptyLabel?: string;
}

interface RowProps {
  node: TreeNode;
  depth: number;
  path: string[];
  onSelect?: (node: TreeNode, path: string[]) => void;
}

function TreeRow({ node, depth, path, onSelect }: RowProps) {
  const [open, setOpen] = useState(false);
  const children = node.children ?? [];
  const foldable = children.length > 0;
  const here = [...path, node.name];

  return (
    <li className="ds-tree-item">
      <div className="ds-tree-row" style={{ paddingLeft: `${depth * 12 + 6}px` }}>
        {foldable ? (
          <button
            className="ds-tree-toggle"
            aria-expanded={open}
            aria-label={`${open ? "Collapse" : "Expand"} ${node.name}`}
            onClick={() => setOpen(!open)}
          >
            {open ? "▾" : "▸"}
          </button>
        ) : (
          <span className="ds-tree-toggle ds-tree-toggle-empty" />
        )}
        <button className="ds-tree-name" onClick={() => onSelect?.(node, here)} title={here.join("/")}>
          {node.name}
        </button>
        <Badge>{node.kind}</Badge>
        {node.detail ? <span className="ds-tree-detail">{node.detail}</span> : null}
      </div>
      {foldable && open ? (
        <ul className="ds-tree-children">
          {children.map((child) => (
            <TreeRow
              key={`${here.join("/")}/${child.name}`}
              node={child}
              depth={depth + 1}
              path={here}
              onSelect={onSelect}
            />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

export function Tree({ nodes, onSelect, emptyLabel = "nothing to show" }: TreeProps) {
  if (nodes.length === 0) {
    return <p className="ds-tree-empty">{emptyLabel}</p>;
  }
  return (
    <ul className="ds-tree" role="tree">
      {nodes.map((node) => (
        <TreeRow key={node.name} node={node} depth={0} path={[]} onSelect={onSelect} />
      ))}
    </ul>
  );
}
