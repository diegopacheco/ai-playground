import { useForm } from "@tanstack/react-form";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { api } from "../api";

const NAME = /^[a-z0-9][a-z0-9-]{0,40}$/;

export function CreateSandboxForm() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const create = useMutation({
    mutationFn: api.createSandbox,
    onSuccess: async ({ name }) => {
      await client.invalidateQueries({ queryKey: ["sandboxes"] });
      await navigate({ to: "/sandboxes/$name", params: { name } });
    },
  });
  const form = useForm({
    defaultValues: { name: "agent-box" },
    onSubmit: async ({ value }) => {
      await create.mutateAsync(value.name);
    },
  });
  return (
    <form
      className="row"
      onSubmit={(e) => {
        e.preventDefault();
        void form.handleSubmit();
      }}
    >
      <form.Field
        name="name"
        validators={{ onChange: ({ value }) => (NAME.test(value) ? undefined : "lowercase letters, digits, dashes") }}
      >
        {(field) => (
          <label className="field">
            <span>Sandbox name</span>
            <input value={field.state.value} onChange={(e) => field.handleChange(e.target.value)} />
            {field.state.meta.errors.length > 0 && <em className="error">{field.state.meta.errors.join(", ")}</em>}
          </label>
        )}
      </form.Field>
      <button type="submit" disabled={create.isPending}>
        {create.isPending ? "Creating..." : "Create sandbox"}
      </button>
      {create.error && <em className="error">{create.error.message}</em>}
    </form>
  );
}
