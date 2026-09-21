interface AskFormProps {
  submit: (form: FormData) => void;
  pending: boolean;
}

export function AskForm({ submit, pending }: AskFormProps) {
  return (
    <form className="ask-form" action={submit}>
      <textarea name="question" placeholder="Ask anything, then ask it again in different words..." rows={3} disabled={pending} />
      <button type="submit" disabled={pending}>
        {pending ? "Asking..." : "Ask"}
      </button>
    </form>
  );
}
