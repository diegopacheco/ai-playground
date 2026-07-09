const cube = document.getElementById('cube');
const pieces = [];
const faces = ['f-top', 'f-bottom', 'f-front', 'f-back', 'f-right', 'f-left'];

for (let x = -1; x <= 1; x++) {
  for (let y = -1; y <= 1; y++) {
    for (let z = -1; z <= 1; z++) {
      const el = document.createElement('div');
      el.className = 'piece';
      faces.forEach(f => {
        const face = document.createElement('div');
        face.className = `face ${f}`;
        if (
          (f === 'f-top' && y > -1) ||
          (f === 'f-bottom' && y < 1) ||
          (f === 'f-front' && z < 1) ||
          (f === 'f-back' && z > -1) ||
          (f === 'f-right' && x < 1) ||
          (f === 'f-left' && x > -1)
        ) {
          face.style.background = '#000';
        }
        el.appendChild(face);
      });
      el.style.transform = `translate3d(${x * 50}px, ${y * 50}px, ${z * 50}px)`;
      cube.appendChild(el);
      pieces.push({ el, x, y, z });
    }
  }
}

let isAnimating = false;
let queue = [];
let moveHistory = [];

function rotateLayer(axis, layer, angle, duration) {
  return new Promise(resolve => {
    isAnimating = true;
    const activePieces = pieces.filter(p => p[axis] === layer);
    const group = document.createElement('div');
    group.className = 'group';
    cube.appendChild(group);
    
    activePieces.forEach(p => group.appendChild(p.el));
    
    group.offsetHeight;
    group.style.transition = `transform ${duration}ms linear`;
    group.style.transform = `rotate${axis.toUpperCase()}(${angle}deg)`;
    
    setTimeout(() => {
      let exactGroupMatrix = new DOMMatrix();
      if (axis === 'x') exactGroupMatrix = exactGroupMatrix.rotate(angle, 0, 0);
      if (axis === 'y') exactGroupMatrix = exactGroupMatrix.rotate(0, angle, 0);
      if (axis === 'z') exactGroupMatrix = exactGroupMatrix.rotate(0, 0, angle);

      activePieces.forEach(p => {
        const m = exactGroupMatrix.multiply(new DOMMatrix(p.el.style.transform));
        
        m.m11 = Math.round(m.m11); m.m12 = Math.round(m.m12); m.m13 = Math.round(m.m13);
        m.m21 = Math.round(m.m21); m.m22 = Math.round(m.m22); m.m23 = Math.round(m.m23);
        m.m31 = Math.round(m.m31); m.m32 = Math.round(m.m32); m.m33 = Math.round(m.m33);
        m.m41 = Math.round(m.m41); m.m42 = Math.round(m.m42); m.m43 = Math.round(m.m43);

        p.el.style.transform = m.toString();
        cube.appendChild(p.el);
        const pt = m.transformPoint(new DOMPoint(0, 0, 0));
        p.x = Math.round(pt.x / 50);
        p.y = Math.round(pt.y / 50);
        p.z = Math.round(pt.z / 50);
      });
      group.remove();
      isAnimating = false;
      resolve();
    }, duration);
  });
}

async function processQueue() {
  if (queue.length === 0 || isAnimating) return;
  const move = queue.shift();
  await rotateLayer(move.axis, move.layer, move.angle, move.duration);
  processQueue();
}

function addMove(axis, layer, angle, duration, record = false) {
  if (record) {
    moveHistory.push({ axis, layer, angle: -angle });
  }
  queue.push({ axis, layer, angle, duration });
  processQueue();
}

const axes = ['x', 'y', 'z'];
const layers = [-1, 0, 1];
const angles = [90, -90];

document.getElementById('scramble').addEventListener('click', () => {
  if (queue.length > 0) return;
  moveHistory = [];
  for (let i = 0; i < 20; i++) {
    const axis = axes[Math.floor(Math.random() * axes.length)];
    const layer = layers[Math.floor(Math.random() * layers.length)];
    const angle = angles[Math.floor(Math.random() * angles.length)];
    addMove(axis, layer, angle, 150, true);
  }
});

document.getElementById('solve').addEventListener('click', () => {
  if (queue.length > 0 || moveHistory.length === 0) return;
  const reverseMoves = [...moveHistory].reverse();
  moveHistory = [];
  reverseMoves.forEach(m => addMove(m.axis, m.layer, m.angle, 150, false));
});

let isDragging = false;
let prevX, prevY;
let rotX = -30, rotY = 45;

document.addEventListener('mousedown', e => {
  if(e.target.tagName === 'BUTTON') return;
  isDragging = true;
  prevX = e.clientX;
  prevY = e.clientY;
});

document.addEventListener('mousemove', e => {
  if (!isDragging) return;
  const deltaX = e.clientX - prevX;
  const deltaY = e.clientY - prevY;
  rotY += deltaX * 0.5;
  rotX -= deltaY * 0.5;
  cube.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg)`;
  prevX = e.clientX;
  prevY = e.clientY;
});

document.addEventListener('mouseup', () => {
  isDragging = false;
});
