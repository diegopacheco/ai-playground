import { useForm } from "@tanstack/react-form";
import { ErrorNote } from "../../../components/ErrorNote";
import { Presets } from "../../../components/Presets";
import { useTriage } from "../hooks";
import { TriageView } from "./TriageView";

const MESSAGES = [
  "My new headphones arrived yesterday, but the left side doesn't work. Can I get my money back?",
  "Where is my order? I ordered it two weeks ago.",
  "Thanks, the new charger works great!",
] as const;

export function TriagePanel() {
  const triage = useTriage();
  const form = useForm({
    defaultValues: { message: "" },
    onSubmit: ({ value }) => triage.mutate(value.message.trim()),
  });

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Support triage</h2>
        <p>One call asks three typed questions about a customer message: a yes/no, a choice and a score.</p>
      </header>
      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          void form.handleSubmit();
        }}
      >
        <form.Field name="message" validators={{ onSubmit: ({ value }) => (value.trim() ? undefined : "Write a message") }}>
          {(field) => (
            <>
              <label htmlFor={field.name}>Customer message</label>
              <textarea id={field.name} rows={3} value={field.state.value} onChange={(e) => field.handleChange(e.target.value)} />
              {field.state.meta.errors.length > 0 ? <p className="error">{field.state.meta.errors.join(", ")}</p> : null}
              <Presets items={MESSAGES} onPick={(text) => field.handleChange(text)} />
            </>
          )}
        </form.Field>
        <button type="submit" className="go" disabled={triage.isPending}>
          {triage.isPending ? "Deciding..." : "Decide"}
        </button>
      </form>
      <ErrorNote error={triage.error} />
      {triage.data ? <TriageView result={triage.data} /> : null}
    </section>
  );
}
