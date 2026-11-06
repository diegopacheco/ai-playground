import { useForm } from "@tanstack/react-form";
import { useMutation } from "@tanstack/react-query";
import type { Agent } from "../agent";
import { Card, ErrorLine, JsonBlock, WithAgent } from "../components";

const MAX_TASK = 200;

function Feedback({ agent }: { agent: Agent }) {
  const endpoint = agent.document.feedback?.endpoint;
  const submit = useMutation({
    mutationFn: (body: { task: string; description: string }) => agent.client.call<{ status: string; task: string }>(endpoint ?? "", { method: "POST", body }),
  });
  const form = useForm({
    defaultValues: { task: "Pay with a gift card", description: "The user wanted to pay a stay with a gift card. Neither the API nor the Company Agent can apply one." },
    onSubmit: ({ value }) => submit.mutate(value),
  });
  if (!endpoint) return <p className="dim">This Company does not ask for feedback.</p>;
  return (
    <Card title="Report a task Poppy could not do (POST feedback.endpoint)">
      <form
        className="stack"
        onSubmit={(event) => {
          event.preventDefault();
          form.handleSubmit();
        }}
      >
        <form.Field name="task" validators={{ onChange: ({ value }) => (value.length > MAX_TASK ? `at most ${MAX_TASK} characters` : undefined) }}>
          {(field) => (
            <label>
              Task
              <input value={field.state.value} onChange={(event) => field.handleChange(event.target.value)} />
              {field.state.meta.errors.length ? <span className="error">{field.state.meta.errors.join(", ")}</span> : null}
            </label>
          )}
        </form.Field>
        <form.Field name="description">
          {(field) => (
            <label>
              Description (no personal data)
              <textarea rows={4} value={field.state.value} onChange={(event) => field.handleChange(event.target.value)} />
            </label>
          )}
        </form.Field>
        <button type="submit">Send feedback</button>
      </form>
      {submit.data ? <JsonBlock value={submit.data} /> : null}
      <ErrorLine error={submit.error} />
    </Card>
  );
}

export function FeedbackPage() {
  return <WithAgent>{(agent) => <Feedback agent={agent} />}</WithAgent>;
}
