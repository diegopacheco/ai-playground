import { useEffect, useRef } from "react";
import { EditorState, type Extension } from "@codemirror/state";
import { EditorView, keymap, placeholder as placeholderExt } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { autocompletion, completionKeymap, type CompletionContext } from "@codemirror/autocomplete";
import { sql } from "@codemirror/lang-sql";
import { json } from "@codemirror/lang-json";
import type { EngineDescriptor } from "@engines/types";
import "./QueryEditor.css";

export interface QueryEditorProps {
  engine: EngineDescriptor;
  value: string;
  completions: string[];
  onChange: (value: string) => void;
  onRun: () => void;
}

function languageExtension(engine: EngineDescriptor): Extension[] {
  if (engine.language === "sql") {
    return [sql()];
  }
  if (engine.language === "json") {
    return [json()];
  }
  return [];
}

export function QueryEditor({ engine, value, completions, onChange, onRun }: QueryEditorProps) {
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
      const word = context.matchBefore(/[\w:/.\-_]*/);
      if (!word || (word.from === word.to && !context.explicit)) {
        return null;
      }
      return {
        from: word.from,
        options: completionsRef.current.map((label) => ({ label, type: "keyword" }))
      };
    };

    const state = EditorState.create({
      doc: value,
      extensions: [
        history(),
        autocompletion({ override: [complete] }),
        placeholderExt(engine.placeholder),
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
        ...languageExtension(engine),
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
  }, [engine.kind]);

  useEffect(() => {
    const editor = view.current;
    if (editor && value !== editor.state.doc.toString()) {
      editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } });
    }
  }, [value]);

  return <div className="query-editor" data-testid="query-editor" ref={host} />;
}
