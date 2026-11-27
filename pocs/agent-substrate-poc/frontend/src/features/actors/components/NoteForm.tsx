import { useForm } from "@tanstack/react-form";
import { useRemember } from "../hooks";

type NoteFormProps = { readonly name: string; readonly onSaved: () => void };

export function NoteForm({ name, onSaved }: NoteFormProps) {
  const remember = useRemember(name);
  const form = useForm({
    defaultValues: { text: "" },
    onSubmit: async ({ value, formApi }) => {
      await remember.mutateAsync(value.text);
      formApi.reset();
      onSaved();
    },
  });
  return (
    <form
      className="create"
      onSubmit={(e) => {
        e.preventDefault();
        void form.handleSubmit();
      }}
    >
      <form.Field name="text" validators={{ onChange: ({ value }) => (value.trim() ? undefined : "Write something") }}>
        {(field) => (
          <input
            className="grow"
            placeholder="tell the agent something to remember (wakes it up if hibernated)"
            value={field.state.value}
            onChange={(e) => field.handleChange(e.target.value)}
          />
        )}
      </form.Field>
      <form.Subscribe selector={(s) => s.canSubmit && s.values.text.trim() !== ""}>
        {(canSubmit) => (
          <button type="submit" className="btn btn-primary" disabled={!canSubmit || remember.isPending}>
            {remember.isPending ? "Sending..." : "Remember"}
          </button>
        )}
      </form.Subscribe>
      {remember.isError && <span className="error">{remember.error.message}</span>}
    </form>
  );
}
