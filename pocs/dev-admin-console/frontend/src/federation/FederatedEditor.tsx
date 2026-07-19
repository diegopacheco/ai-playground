import { useEffect, useRef } from "react";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, placeholder as placeholderExt } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { autocompletion, completionKeymap, type CompletionContext } from "@codemirror/autocomplete";
import { sql } from "@codemirror/lang-sql";
import "../console/QueryEditor.css";

export interface FederatedCompletion {
  label: string;
  detail?: string;
  type?: string;
}

export interface FederatedEditorProps {
  value: string;
  completions: FederatedCompletion[];
  onChange: (value: string) => void;
  onRun: () => void;
}

const KEYWORDS: FederatedCompletion[] = [
  { label: "SELECT", type: "keyword" },
  { label: "FROM", type: "keyword" },
  { label: "JOIN", type: "keyword" },
  { label: "LEFT JOIN", type: "keyword" },
  { label: "INNER JOIN", type: "keyword" },
  { label: "ON", type: "keyword" },
  { label: "WHERE", type: "keyword" },
  { label: "LIMIT", type: "keyword" }
];

export function FederatedEditor({ value, completions, onChange, onRun }: FederatedEditorProps) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const runRef = useRef(onRun);
  const changeRef = useRef(onChange);
  const completionsRef = useRef(completions);

  runRef.current = onRun;
  changeRef.current = onChange;
  completionsRef.current = completions;

  useEffect(() => {
    if (!host.current) {
      return;
    }
    const complete = (context: CompletionContext) => {
      const word = context.matchBefore(/[\w.\-_]*/);
      if (!word || (word.from === word.to && !context.explicit)) {
        return null;
      }
      return {
        from: word.from,
        options: [...KEYWORDS, ...completionsRef.current].map((option) => ({
          label: option.label,
          detail: option.detail,
          type: option.type ?? "variable"
        }))
      };
    };

    const state = EditorState.create({
      doc: value,
      extensions: [
        history(),
        autocompletion({ override: [complete], activateOnTyping: true }),
        placeholderExt(
          "SELECT a.email, b.name\nFROM demo-postgres.customers a\nJOIN demo-elasticsearch.products b ON a.id = b._id\nLIMIT 25"
        ),
        keymap.of([
          ...["Mod-Enter", "Cmd-Enter", "Ctrl-Enter"].map((key) => ({
            key,
            preventDefault: true,
            run: () => {
              runRef.current();
              return true;
            }
          })),
          ...completionKeymap,
          ...historyKeymap,
          ...defaultKeymap
        ]),
        sql(),
        EditorView.lineWrapping,
        EditorView.updateListener.of((update) => {
          if (update.docChanged) {
            changeRef.current(update.state.doc.toString());
          }
        })
      ]
    });

    const editor = new EditorView({ state, parent: host.current });
    view.current = editor;
    return () => {
      editor.destroy();
      view.current = null;
    };
  }, []);

  useEffect(() => {
    const editor = view.current;
    if (editor && value !== editor.state.doc.toString()) {
      editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } });
    }
  }, [value]);

  return <div className="query-editor fed-cm" data-testid="federated-editor" ref={host} />;
}
