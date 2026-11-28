import { sound } from "../../../lib/sound";
import { PAINT_SWATCHES, SURFACE_LABELS } from "../../house/catalog";
import type { FurnitureItem, SelectionTarget, SurfaceId } from "../../house/types";

type ColorPaletteBarProps = {
  readonly selection: SelectionTarget;
  readonly furniture: readonly FurnitureItem[];
  readonly onPaintFurniture: (id: string, color: string) => void;
  readonly onCycleStyle: (id: string) => void;
  readonly onPaintSurface: (id: SurfaceId, color: string) => void;
  readonly onOpenKitchen: () => void;
  readonly onOpenPhone: () => void;
  readonly onClose: () => void;
};

export function ColorPaletteBar({
  selection,
  furniture,
  onPaintFurniture,
  onCycleStyle,
  onPaintSurface,
  onOpenKitchen,
  onOpenPhone,
  onClose,
}: ColorPaletteBarProps) {
  if (selection.kind === "character") return null;

  const selectedFurniture =
    selection.kind === "furniture"
      ? furniture.find((f) => f.id === selection.id) ?? null
      : null;

  const title =
    selection.kind === "surface"
      ? SURFACE_LABELS[selection.id]
      : selectedFurniture?.label ?? "Paint";

  const handlePickColor = (hex: string) => {
    sound.paint();
    if (selection.kind === "furniture") {
      onPaintFurniture(selection.id, hex);
    } else {
      onPaintSurface(selection.id, hex);
    }
  };

  return (
    <div className="palette-bar" role="region" aria-label="Color and Style Picker">
      <div className="palette-header">
        <span className="palette-title">🎨 {title}</span>
        <div className="palette-actions">
          {selectedFurniture && (
            <button
              type="button"
              className="toy-chip"
              onClick={() => {
                sound.paint();
                onCycleStyle(selectedFurniture.id);
              }}
            >
              ✨ Style
            </button>
          )}
          {selectedFurniture?.kind === "stove" && (
            <button type="button" className="toy-chip toy-chip-warm" onClick={onOpenKitchen}>
              🍳 Cook!
            </button>
          )}
          {selectedFurniture?.kind === "phoneTable" && (
            <button type="button" className="toy-chip toy-chip-warm" onClick={onOpenPhone}>
              📞 Call!
            </button>
          )}
          <button type="button" className="toy-close-mini" onClick={onClose} aria-label="Close colors">
            ✕
          </button>
        </div>
      </div>
      <div className="swatch-row">
        {PAINT_SWATCHES.map((swatch) => (
          <button
            key={swatch.hex}
            type="button"
            className="swatch-btn"
            style={{ backgroundColor: swatch.hex }}
            onClick={() => handlePickColor(swatch.hex)}
            aria-label={`Paint ${swatch.name}`}
            title={swatch.name}
          />
        ))}
      </div>
    </div>
  );
}
