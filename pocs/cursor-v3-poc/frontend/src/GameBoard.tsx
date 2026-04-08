import { useRef, useEffect } from "react";
import { Board, Piece, Position, BOARD_WIDTH, BOARD_HEIGHT, CELL_SIZE, Theme } from "./types";

interface Props {
  board: Board;
  currentPiece: Piece | null;
  currentPos: Position;
  theme: Theme;
}

export default function GameBoard({ board, currentPiece, currentPos, theme }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = theme.boardBg;
    ctx.fillRect(0, 0, BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE);

    for (let y = 0; y < BOARD_HEIGHT; y++) {
      for (let x = 0; x < BOARD_WIDTH; x++) {
        ctx.strokeStyle = theme.gridLine;
        ctx.strokeRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        if (board[y][x]) {
          ctx.fillStyle = board[y][x]!;
          ctx.fillRect(x * CELL_SIZE + 1, y * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2);
          ctx.fillStyle = "rgba(255,255,255,0.15)";
          ctx.fillRect(x * CELL_SIZE + 1, y * CELL_SIZE + 1, CELL_SIZE - 2, 4);
        }
      }
    }

    if (currentPiece) {
      for (let r = 0; r < currentPiece.shape.length; r++) {
        for (let c = 0; c < currentPiece.shape[r].length; c++) {
          if (!currentPiece.shape[r][c]) continue;
          const px = (currentPos.x + c) * CELL_SIZE;
          const py = (currentPos.y + r) * CELL_SIZE;
          if (currentPos.y + r < 0) continue;
          ctx.fillStyle = currentPiece.color;
          ctx.fillRect(px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2);
          ctx.fillStyle = "rgba(255,255,255,0.2)";
          ctx.fillRect(px + 1, py + 1, CELL_SIZE - 2, 4);
        }
      }
    }
  }, [board, currentPiece, currentPos, theme]);

  return (
    <canvas
      ref={canvasRef}
      width={BOARD_WIDTH * CELL_SIZE}
      height={BOARD_HEIGHT * CELL_SIZE}
      style={{ border: `2px solid ${theme.accent}`, borderRadius: 4 }}
    />
  );
}
