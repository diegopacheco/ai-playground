import { useForm } from "@tanstack/react-form";
import { useCreateActor } from "../hooks";
import { nameError } from "../rules";

export function CreateActorForm() {
  const create = useCreateActor();
  const form = useForm({
    defaultValues: { name: "" },
    onSubmit: async ({ value, formApi }) => {
      await create.mutateAsync(value.name);
      formApi.reset();
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
      <form.Field name="name" validators={{ onChange: ({ value }) => nameError(value) }}>
        {(field) => (
          <div className="field">
            <input
              placeholder="new agent name, e.g. alice"
              value={field.state.value}
              onChange={(e) => field.handleChange(e.target.value)}
            />
            {field.state.meta.isTouched && field.state.meta.errors.length > 0 && (
              <span className="error">{field.state.meta.errors.join(", ")}</span>
            )}
          </div>
        )}
      </form.Field>
      <form.Subscribe selector={(s) => s.canSubmit}>
        {(canSubmit) => (
          <button type="submit" className="btn btn-primary" disabled={!canSubmit || create.isPending}>
            {create.isPending ? "Creating..." : "Create actor"}
          </button>
        )}
      </form.Subscribe>
      {create.isError && <span className="error">{create.error.message}</span>}
    </form>
  );
}
