const MAP = {
  ArrowLeft: 'left',
  KeyA: 'left',
  ArrowRight: 'right',
  KeyD: 'right',
  ArrowUp: 'tuck',
  KeyW: 'tuck',
  ArrowDown: 'brake',
  KeyS: 'brake',
  Space: 'jump',
  ShiftLeft: 'grab',
  ShiftRight: 'grab',
  KeyJ: 'grab'
};

export class Input {
  constructor() {
    this.down = new Set();
    this.state = { steer: 0, tuck: false, brake: false, jump: false, grab: false };
    this.onKey = null;
    window.addEventListener('keydown', (e) => {
      const a = MAP[e.code];
      if (a) {
        this.down.add(a);
        e.preventDefault();
      }
      if (this.onKey) this.onKey(e.code);
    });
    window.addEventListener('keyup', (e) => {
      const a = MAP[e.code];
      if (a) {
        this.down.delete(a);
        e.preventDefault();
      }
    });
    window.addEventListener('blur', () => this.down.clear());
  }

  read() {
    const s = this.state;
    s.steer = (this.down.has('right') ? 1 : 0) - (this.down.has('left') ? 1 : 0);
    s.tuck = this.down.has('tuck');
    s.brake = this.down.has('brake');
    s.jump = this.down.has('jump');
    s.grab = this.down.has('grab');
    return s;
  }
}
