"use client";

import { useEffect, useRef } from "react";

export interface PixelCanvasProps {
  readonly width: number;
  readonly height: number;
  readonly className?: string;
  readonly draw: (ctx: CanvasRenderingContext2D, frame: number) => void;
}

export function PixelCanvas({ width, height, className, draw }: PixelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    let handle = 0;
    let frame = 0;
    const loop = () => {
      const ctx = canvasRef.current?.getContext("2d");
      if (ctx) drawRef.current(ctx, frame);
      frame += 1;
      handle = requestAnimationFrame(loop);
    };
    handle = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(handle);
  }, []);

  return <canvas ref={canvasRef} width={width} height={height} className={className} />;
}
