import { useRef, useEffect } from "react";
import { Piece, Theme, CELL_SIZE } from "./types";

interface Props {
  piece: Piece | null;
  theme: Theme;
}

export default function NextPiece({ piece, theme }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = 5 * CELL_SIZE;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = theme.boardBg;
    ctx.fillRect(0, 0, size, size);

    if (!piece) return;
    const offsetX = Math.floor((5 - piece.shape[0].length) / 2);
    const offsetY = Math.floor((5 - piece.shape.length) / 2);

    for (let r = 0; r < piece.shape.length; r++) {
      for (let c = 0; c < piece.shape[r].length; c++) {
        if (!piece.shape[r][c]) continue;
        const px = (offsetX + c) * CELL_SIZE;
        const py = (offsetY + r) * CELL_SIZE;
        ctx.fillStyle = piece.color;
        ctx.fillRect(px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2);
      }
    }
  }, [piece, theme, size]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{ border: `1px solid ${theme.gridLine}`, borderRadius: 4 }}
    />
  );
}
