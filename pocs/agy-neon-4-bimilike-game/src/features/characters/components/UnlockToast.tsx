import { getCharacterSpec } from "../catalog";
import type { CharacterId } from "../types";

type UnlockToastProps = {
  readonly characterId: CharacterId;
  readonly onDismiss: () => void;
};

export function UnlockToast({ characterId, onDismiss }: UnlockToastProps) {
  const spec = getCharacterSpec(characterId);
  return (
    <div className="unlock-toast" role="status" aria-live="polite" onClick={onDismiss}>
      <span className="unlock-star">🌟</span>
      <span className="unlock-emoji">{spec.emoji}</span>
      <div className="unlock-copy">
        <strong>New Friend: {spec.name}!</strong>
        <span>{spec.greeting}</span>
      </div>
    </div>
  );
}
