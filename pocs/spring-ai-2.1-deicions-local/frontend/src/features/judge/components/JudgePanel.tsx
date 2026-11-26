import { useForm } from "@tanstack/react-form";
import { ErrorNote } from "../../../components/ErrorNote";
import { Presets } from "../../../components/Presets";
import { useJudge } from "../hooks";
import { JudgeView } from "./JudgeView";

const ANSWERS = [
  "It is sunny in Paris, with a temperature of 15°C.",
  "It is sunny in Paris, with a temperature of -255°C.",
  "I really like trains.",
] as const;

const required = ({ value }: { readonly value: string }) => (value.trim() ? undefined : "Required");

export function JudgePanel() {
  const judge = useJudge();
  const form = useForm({
    defaultValues: { question: "What is the current weather in Paris?", answer: "" },
    onSubmit: ({ value }) => judge.mutate({ question: value.question.trim(), answer: value.answer.trim() }),
  });

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Judge</h2>
        <p>Two criteria with their own thresholds, evaluated in one call. A failure comes with feedback for a second try.</p>
      </header>
      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          void form.handleSubmit();
        }}
      >
        <form.Field name="question" validators={{ onSubmit: required }}>
          {(field) => (
            <>
              <label htmlFor={field.name}>User question</label>
              <input id={field.name} value={field.state.value} onChange={(e) => field.handleChange(e.target.value)} />
              {field.state.meta.errors.length > 0 ? <p className="error">{field.state.meta.errors.join(", ")}</p> : null}
            </>
          )}
        </form.Field>
        <form.Field name="answer" validators={{ onSubmit: required }}>
          {(field) => (
            <>
              <label htmlFor={field.name}>Assistant answer</label>
              <textarea id={field.name} rows={2} value={field.state.value} onChange={(e) => field.handleChange(e.target.value)} />
              {field.state.meta.errors.length > 0 ? <p className="error">{field.state.meta.errors.join(", ")}</p> : null}
              <Presets items={ANSWERS} onPick={(text) => field.handleChange(text)} />
            </>
          )}
        </form.Field>
        <button type="submit" className="go" disabled={judge.isPending}>
          {judge.isPending ? "Judging..." : "Judge"}
        </button>
      </form>
      <ErrorNote error={judge.error} />
      {judge.data ? <JudgeView result={judge.data} /> : null}
    </section>
  );
}
