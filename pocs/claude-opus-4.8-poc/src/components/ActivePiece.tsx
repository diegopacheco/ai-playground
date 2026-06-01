import { useStore } from "@tanstack/react-store";
import { store } from "../game/store";
import { colorOf } from "../game/pieces";
import { Block } from "./Block";

export function ActivePiece() {
  const piece = useStore(store, (s) => s.piece);
  if (!piece) return null;
  return (
    <group>
      {piece.cells.map((cell, i) => (
        <Block
          key={i}
          position={[piece.pos.x + cell.x + 0.5, piece.pos.y + cell.y + 0.5, piece.pos.z + cell.z + 0.5]}
          color={colorOf(piece.color)}
          emissive={0.55}
        />
      ))}
    </group>
  );
}
