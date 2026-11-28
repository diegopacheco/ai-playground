import { useState } from "react";
import { sound } from "../../../lib/sound";
import { cookRecipe, getIngredientSpec, INGREDIENTS } from "../recipes";
import type { DishSpec, IngredientId } from "../types";

type KitchenSheetProps = {
  readonly onCooked: (dish: DishSpec) => void;
  readonly onClose: () => void;
};

export function KitchenSheet({ onCooked, onClose }: KitchenSheetProps) {
  const [bowl, setBowl] = useState<readonly IngredientId[]>([]);
  const [cooking, setCooking] = useState(false);
  const [readyDish, setReadyDish] = useState<DishSpec | null>(null);

  const toggleIngredient = (id: IngredientId) => {
    if (cooking) return;
    sound.bubble();
    setReadyDish(null);
    setBowl((prev) => {
      if (prev.length >= 3) {
        return [...prev.slice(1), id];
      }
      return [...prev, id];
    });
  };

  const handleCook = () => {
    if (bowl.length === 0 || cooking) return;
    setCooking(true);
    sound.bubble();
    const result = cookRecipe(bowl);
    window.setTimeout(() => {
      sound.cookSuccess();
      setCooking(false);
      setReadyDish(result);
      setBowl([]);
      onCooked(result);
    }, 650);
  };

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Magic Kitchen">
      <div className="toy-modal kitchen-modal">
        <div className="modal-top">
          <h2 className="modal-title">🍳 Magic Kitchen</h2>
          <button type="button" className="toy-close" onClick={onClose} aria-label="Close kitchen">
            ✕
          </button>
        </div>

        <div className={`mixing-bowl ${cooking ? "bowl-stirring" : ""}`}>
          {readyDish ? (
            <div className="ready-dish-banner">
              <span className="ready-emoji">{readyDish.emoji}</span>
              <span className="ready-text">Yum! {readyDish.name} is on the table!</span>
            </div>
          ) : bowl.length === 0 ? (
            <div className="bowl-hint">Tap yummy food to put in the pot!</div>
          ) : (
            <div className="bowl-items">
              {bowl.map((id, idx) => {
                const spec = getIngredientSpec(id);
                return (
                  <span key={`${id}-${idx}`} className="bowl-pill">
                    {spec.emoji}
                  </span>
                );
              })}
            </div>
          )}
        </div>

        <div className="ingredient-grid">
          {INGREDIENTS.map((item) => (
            <button
              key={item.id}
              type="button"
              className="ingredient-card"
              style={{ borderColor: item.color }}
              onClick={() => toggleIngredient(item.id)}
            >
              <span className="ingredient-emoji">{item.emoji}</span>
              <span className="ingredient-name">{item.name}</span>
            </button>
          ))}
        </div>

        <div className="kitchen-footer">
          <button
            type="button"
            className="toy-big-btn toy-btn-cook"
            disabled={bowl.length === 0 || cooking}
            onClick={handleCook}
          >
            {cooking ? "✨ Cooking..." : "🥄 Mix & Bake!"}
          </button>
        </div>
      </div>
    </div>
  );
}
