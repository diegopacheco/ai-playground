import { useActorAction } from "../hooks";
import { allowedActions } from "../rules";
import type { ActorState } from "../types";

type ActionButtonsProps = { readonly name: string; readonly state: ActorState };

export function ActionButtons({ name, state }: ActionButtonsProps) {
  const action = useActorAction();
  const busy = action.isPending && action.variables?.name === name;
  return (
    <div className="actions">
      {allowedActions(state).map((a) => (
        <button key={a} type="button" className={`btn btn-${a}`} disabled={busy} onClick={() => action.mutate({ name, action: a })}>
          {busy && action.variables?.action === a ? "..." : a}
        </button>
      ))}
      {action.isError && action.variables?.name === name && <span className="error">{action.error.message}</span>}
    </div>
  );
}
