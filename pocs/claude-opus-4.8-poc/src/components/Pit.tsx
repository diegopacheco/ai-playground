import * as THREE from "three";
import { useStore } from "@tanstack/react-store";
import { store, ghostPos } from "../game/store";
import { WIDTH, DEPTH, HEIGHT } from "../game/types";
import { colorOf } from "../game/pieces";
import { Block } from "./Block";

const FRAME = new THREE.EdgesGeometry(new THREE.BoxGeometry(WIDTH, HEIGHT, DEPTH));

export function Pit() {
  const board = useStore(store, (s) => s.board);
  const piece = useStore(store, (s) => s.piece);
  const ghost = piece ? ghostPos(store.state) : null;

  const blocks = [];
  for (let i = 0; i < board.length; i++) {
    const c = board[i];
    if (c === 0) continue;
    const y = Math.floor(i / (WIDTH * DEPTH));
    const rem = i % (WIDTH * DEPTH);
    const z = Math.floor(rem / WIDTH);
    const x = rem % WIDTH;
    blocks.push(
      <Block key={i} position={[x + 0.5, y + 0.5, z + 0.5]} color={colorOf(c)} />,
    );
  }

  return (
    <group>
      <lineSegments geometry={FRAME} position={[WIDTH / 2, HEIGHT / 2, DEPTH / 2]}>
        <lineBasicMaterial color="#334155" />
      </lineSegments>
      <gridHelper
        args={[Math.max(WIDTH, DEPTH), Math.max(WIDTH, DEPTH), "#1e293b", "#1e293b"]}
        position={[WIDTH / 2, 0, DEPTH / 2]}
      />
      {blocks}
      {piece && ghost
        ? piece.cells.map((cell, i) => (
            <Block
              key={`g${i}`}
              position={[ghost.x + cell.x + 0.5, ghost.y + cell.y + 0.5, ghost.z + cell.z + 0.5]}
              color={colorOf(piece.color)}
              opacity={0.16}
              edge="#475569"
            />
          ))
        : null}
    </group>
  );
}
