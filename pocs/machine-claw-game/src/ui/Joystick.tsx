import { type PointerEvent, useState } from "react";

type JoystickProps = {
  readonly disabled: boolean;
  readonly onMove: (x: number, y: number) => void;
};

type Offset = { readonly x: number; readonly y: number };

const CENTER: Offset = { x: 0, y: 0 };

export function Joystick({ disabled, onMove }: JoystickProps) {
  const [offset, setOffset] = useState<Offset>(CENTER);

  function track(event: PointerEvent<HTMLDivElement>): void {
    const box = event.currentTarget.getBoundingClientRect();
    const reach = box.width / 2;
    const rawX = (event.clientX - box.left - reach) / reach;
    const rawY = (event.clientY - box.top - reach) / reach;
    const length = Math.max(1, Math.hypot(rawX, rawY));
    const next = { x: rawX / length, y: rawY / length };
    setOffset(next);
    onMove(next.x, next.y);
  }

  function press(event: PointerEvent<HTMLDivElement>): void {
    event.currentTarget.setPointerCapture(event.pointerId);
    track(event);
  }

  function release(): void {
    setOffset(CENTER);
    onMove(0, 0);
  }

  return (
    <div
      className={disabled ? "joystick joystick-off" : "joystick"}
      aria-label="Move claw"
      onPointerDown={press}
      onPointerMove={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) track(event);
      }}
      onPointerUp={release}
      onPointerCancel={release}
    >
      <div className="joystick-knob" style={{ transform: `translate(${offset.x * 34}px, ${offset.y * 34}px)` }} />
    </div>
  );
}
