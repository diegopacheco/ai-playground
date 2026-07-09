const cube = document.querySelector("#cube");
const scrambleButton = document.querySelector("#scramble");
const solveButton = document.querySelector("#solve");
const statusText = document.querySelector("#status");

const faces = ["u", "d", "l", "r", "f", "b"];
const colors = {
  u: "var(--white)",
  d: "var(--yellow)",
  l: "var(--orange)",
  r: "var(--red)",
  f: "var(--green)",
  b: "var(--blue)"
};

const moves = [
  { name: "R", axis: "x", layer: 1, dir: 1 },
  { name: "L", axis: "x", layer: -1, dir: -1 },
  { name: "U", axis: "y", layer: -1, dir: 1 },
  { name: "D", axis: "y", layer: 1, dir: -1 },
  { name: "F", axis: "z", layer: 1, dir: 1 },
  { name: "B", axis: "z", layer: -1, dir: -1 }
];

let cubies = [];
let history = [];
let busy = false;

function stickerSet(x, y, z) {
  return {
    u: y === -1 ? colors.u : null,
    d: y === 1 ? colors.d : null,
    l: x === -1 ? colors.l : null,
    r: x === 1 ? colors.r : null,
    f: z === 1 ? colors.f : null,
    b: z === -1 ? colors.b : null
  };
}

function resetCube() {
  cubies = [];
  let id = 0;
  for (let x = -1; x <= 1; x += 1) {
    for (let y = -1; y <= 1; y += 1) {
      for (let z = -1; z <= 1; z += 1) {
        cubies.push({ id: id, x: x, y: y, z: z, stickers: stickerSet(x, y, z) });
        id += 1;
      }
    }
  }
}

function baseTransform(cubie) {
  return `translate3d(calc(${cubie.x} * var(--cubie)), calc(${cubie.y} * var(--cubie)), calc(${cubie.z} * var(--cubie)))`;
}

function turnTransform(move, degrees) {
  if (move.axis === "x") {
    return `rotateX(${degrees}deg)`;
  }
  if (move.axis === "y") {
    return `rotateY(${degrees}deg)`;
  }
  return `rotateZ(${degrees}deg)`;
}

function render() {
  cube.innerHTML = "";
  cubies.forEach((cubie) => {
    const piece = document.createElement("div");
    piece.className = "cubie";
    piece.dataset.id = cubie.id;
    piece.style.transform = baseTransform(cubie);
    faces.forEach((face) => {
      const side = document.createElement("div");
      side.className = `face ${face}`;
      side.style.background = cubie.stickers[face] || "var(--plastic)";
      piece.appendChild(side);
    });
    cube.appendChild(piece);
  });
}

function rotateVector(axis, dir, x, y, z) {
  if (axis === "x") {
    return { x: x, y: -dir * z, z: dir * y };
  }
  if (axis === "y") {
    return { x: dir * z, y: y, z: -dir * x };
  }
  return { x: -dir * y, y: dir * x, z: z };
}

function faceVector(face) {
  return {
    u: { x: 0, y: -1, z: 0 },
    d: { x: 0, y: 1, z: 0 },
    l: { x: -1, y: 0, z: 0 },
    r: { x: 1, y: 0, z: 0 },
    f: { x: 0, y: 0, z: 1 },
    b: { x: 0, y: 0, z: -1 }
  }[face];
}

function vectorFace(vector) {
  if (vector.y === -1) {
    return "u";
  }
  if (vector.y === 1) {
    return "d";
  }
  if (vector.x === -1) {
    return "l";
  }
  if (vector.x === 1) {
    return "r";
  }
  if (vector.z === 1) {
    return "f";
  }
  return "b";
}

function applyMove(move) {
  cubies.forEach((cubie) => {
    if (cubie[move.axis] !== move.layer) {
      return;
    }
    const next = rotateVector(move.axis, move.dir, cubie.x, cubie.y, cubie.z);
    const nextStickers = { u: null, d: null, l: null, r: null, f: null, b: null };
    faces.forEach((face) => {
      const color = cubie.stickers[face];
      if (!color) {
        return;
      }
      const vector = faceVector(face);
      const rotated = rotateVector(move.axis, move.dir, vector.x, vector.y, vector.z);
      nextStickers[vectorFace(rotated)] = color;
    });
    cubie.x = next.x;
    cubie.y = next.y;
    cubie.z = next.z;
    cubie.stickers = nextStickers;
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function animateMove(move, remember) {
  busy = true;
  setButtons();
  statusText.textContent = move.name;
  const selected = [...document.querySelectorAll(".cubie")].filter((piece) => {
    const cubie = cubies.find((item) => item.id === Number(piece.dataset.id));
    return cubie[move.axis] === move.layer;
  });
  await sleep(25);
  selected.forEach((piece) => {
    const cubie = cubies.find((item) => item.id === Number(piece.dataset.id));
    piece.classList.add("turning");
    piece.style.transform = `${turnTransform(move, move.dir * 90)} ${baseTransform(cubie)}`;
  });
  await sleep(620);
  applyMove(move);
  if (remember) {
    history.push(move);
  }
  render();
  busy = false;
  setButtons();
}

function inverse(move) {
  return { name: `${move.name}'`, axis: move.axis, layer: move.layer, dir: move.dir * -1 };
}

function setButtons() {
  scrambleButton.disabled = busy;
  solveButton.disabled = busy || history.length === 0;
}

function randomMove(last) {
  const options = moves.filter((move) => !last || move.axis !== last.axis || move.layer !== last.layer);
  return options[Math.floor(Math.random() * options.length)];
}

async function scramble() {
  if (busy) {
    return;
  }
  let last = null;
  for (let index = 0; index < 18; index += 1) {
    const move = randomMove(last);
    last = move;
    await animateMove(move, true);
  }
  statusText.textContent = "Scrambled";
}

async function solve() {
  if (busy || history.length === 0) {
    return;
  }
  while (history.length > 0) {
    await animateMove(inverse(history.pop()), false);
  }
  statusText.textContent = "Solved";
}

scrambleButton.addEventListener("click", scramble);
solveButton.addEventListener("click", solve);

resetCube();
render();
setButtons();
