import * as THREE from "three";
import "./style.css";

const canvas = document.querySelector("#ocean-canvas");
const sceneWrap = document.querySelector("#scene-wrap");
const feedButton = document.querySelector("#feed-button");
const sleepButton = document.querySelector("#sleep-button");
const petButton = document.querySelector("#pet-button");
const swimButton = document.querySelector("#swim-button");
const soundToggle = document.querySelector("#sound-toggle");
const sleepLabel = document.querySelector("#sleep-label");
const swimLabel = document.querySelector("#swim-label");
const speech = document.querySelector("#speech");
const achievement = document.querySelector("#achievement");
const weightValue = document.querySelector("#weight-value");
const heartLayer = document.querySelector("#heart-layer");
const confettiLayer = document.querySelector("#confetti-layer");
const dayNumber = document.querySelector("#day-number");

const ui = {
  food: {
    bar: document.querySelector("#food-bar"),
    value: document.querySelector("#food-value")
  },
  sleep: {
    bar: document.querySelector("#sleep-bar"),
    value: document.querySelector("#sleep-value")
  },
  happy: {
    bar: document.querySelector("#happy-bar"),
    value: document.querySelector("#happy-value")
  }
};

const state = {
  food: 78,
  sleep: 64,
  happy: 82,
  feeds: 0,
  sleeping: false,
  action: "idle",
  actionStarted: 0,
  swimDuration: 0,
  achievement: false,
  elapsedCare: 0,
  elapsedDay: 0,
  day: Number(localStorage.getItem("seagotchi-day") ?? 1),
  soundEnabled: localStorage.getItem("seagotchi-sound") !== "off",
  ambientEvent: null,
  ambientStarted: 0,
  ambientIndex: 0,
  nextAmbientAt: performance.now() / 1000 + 12 + Math.random() * 8
};

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  alpha: false,
  powerPreference: "high-performance"
});
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.BasicShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color("#7bd1df");
scene.fog = new THREE.Fog("#7bd1df", 13, 34);

const camera = new THREE.OrthographicCamera(-7, 7, 4.5, -4.5, 0.1, 60);
camera.position.set(9, 6.5, 12);
camera.lookAt(0, -0.15, 0);

scene.add(new THREE.HemisphereLight("#fff3c2", "#074f6d", 3.2));

const sunLight = new THREE.DirectionalLight("#fff1b0", 4.2);
sunLight.position.set(-5, 10, 7);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(1024, 1024);
sunLight.shadow.camera.left = -8;
sunLight.shadow.camera.right = 8;
sunLight.shadow.camera.top = 8;
sunLight.shadow.camera.bottom = -8;
scene.add(sunLight);

const colors = {
  rock: new THREE.MeshStandardMaterial({ color: "#675f5b", roughness: 1, flatShading: true }),
  rockLight: new THREE.MeshStandardMaterial({ color: "#8b7c70", roughness: 1, flatShading: true }),
  moss: new THREE.MeshStandardMaterial({ color: "#6f9c65", roughness: 1, flatShading: true }),
  fur: new THREE.MeshStandardMaterial({ color: "#6c4637", roughness: 0.92, flatShading: true }),
  furLight: new THREE.MeshStandardMaterial({ color: "#a7775e", roughness: 0.9, flatShading: true }),
  dark: new THREE.MeshStandardMaterial({ color: "#101923", roughness: 1, flatShading: true }),
  eye: new THREE.MeshStandardMaterial({ color: "#fff5cf", roughness: 1, flatShading: true }),
  flipper: new THREE.MeshStandardMaterial({ color: "#56352f", roughness: 1, flatShading: true }),
  fish: new THREE.MeshStandardMaterial({ color: "#f0bc49", roughness: 0.7, flatShading: true }),
  fishFin: new THREE.MeshStandardMaterial({ color: "#ff6b4e", roughness: 0.8, flatShading: true }),
  foam: new THREE.MeshBasicMaterial({ color: "#e2fff3" }),
  poop: new THREE.MeshStandardMaterial({ color: "#493126", roughness: 1, flatShading: true }),
  stink: new THREE.MeshStandardMaterial({
    color: "#a7d65d",
    roughness: 1,
    flatShading: true,
    transparent: true,
    opacity: 0.72
  })
};

const makeMesh = (geometry, material, position, scale, rotation) => {
  const item = new THREE.Mesh(geometry, material);
  item.position.set(...position);
  item.scale.set(...scale);
  item.rotation.set(...rotation);
  item.castShadow = true;
  item.receiveShadow = true;
  return item;
};

const world = new THREE.Group();
scene.add(world);

const oceanGeometry = new THREE.PlaneGeometry(44, 32, 54, 42);
const oceanMaterial = new THREE.ShaderMaterial({
  uniforms: {
    time: { value: 0 },
    deepColor: { value: new THREE.Color("#06496d") },
    lightColor: { value: new THREE.Color("#18a6ba") },
    foamColor: { value: new THREE.Color("#d8f7de") }
  },
  vertexShader: `
    uniform float time;
    varying float waveHeight;
    varying vec2 waterUv;
    void main() {
      vec3 point = position;
      float waveA = sin(point.x * 0.72 + time * 1.65) * 0.24;
      float waveB = sin(point.y * 1.08 - time * 1.15) * 0.14;
      float waveC = sin((point.x + point.y) * 1.6 + time * 0.85) * 0.06;
      point.z += waveA + waveB + waveC;
      waveHeight = point.z;
      waterUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(point, 1.0);
    }
  `,
  fragmentShader: `
    uniform vec3 deepColor;
    uniform vec3 lightColor;
    uniform vec3 foamColor;
    varying float waveHeight;
    varying vec2 waterUv;
    void main() {
      float band = floor((waveHeight + 0.45) * 8.0) / 8.0;
      float shimmer = step(0.93, sin(waterUv.x * 220.0 + waterUv.y * 80.0));
      vec3 color = mix(deepColor, lightColor, clamp(band + 0.35, 0.0, 1.0));
      color = mix(color, foamColor, shimmer * step(0.12, waveHeight) * 0.42);
      gl_FragColor = vec4(color, 1.0);
    }
  `
});
const ocean = new THREE.Mesh(oceanGeometry, oceanMaterial);
ocean.rotation.x = -Math.PI / 2;
ocean.position.y = -1.12;
ocean.receiveShadow = true;
world.add(ocean);

const sun = makeMesh(
  new THREE.CircleGeometry(2.1, 16),
  new THREE.MeshBasicMaterial({ color: "#ffd25a" }),
  [-5.7, 4.4, -10],
  [1, 1, 1],
  [0, 0, 0]
);
world.add(sun);

const distantRockMaterial = new THREE.MeshStandardMaterial({
  color: "#406a6c",
  roughness: 1,
  flatShading: true
});

for (let index = 0; index < 7; index += 1) {
  const distantRock = makeMesh(
    new THREE.DodecahedronGeometry(1, 0),
    distantRockMaterial,
    [-10 + index * 3.7, -0.65, -8.5 - (index % 2) * 1.2],
    [2.3, 0.5 + (index % 3) * 0.15, 1.2],
    [0, index * 0.3, 0]
  );
  distantRock.castShadow = false;
  world.add(distantRock);
}

const rockGroup = new THREE.Group();
rockGroup.position.set(0, -0.42, 0);
world.add(rockGroup);

const rock = makeMesh(
  new THREE.DodecahedronGeometry(2.25, 1),
  colors.rock,
  [0, 0, 0],
  [1.8, 0.48, 1.2],
  [0, 0.2, -0.03]
);
rockGroup.add(rock);

const rockTop = makeMesh(
  new THREE.DodecahedronGeometry(1.85, 1),
  colors.rockLight,
  [-0.25, 0.48, 0.04],
  [1.75, 0.24, 1.03],
  [0, -0.13, 0]
);
rockGroup.add(rockTop);

for (let index = 0; index < 6; index += 1) {
  const moss = makeMesh(
    new THREE.IcosahedronGeometry(0.42, 0),
    colors.moss,
    [-2.8 + index * 1.08, 0.65 + (index % 2) * 0.05, 0.6 - (index % 3) * 0.35],
    [1.3, 0.12, 0.65],
    [0, index * 0.8, 0]
  );
  moss.castShadow = false;
  rockGroup.add(moss);
}

const sealRoot = new THREE.Group();
sealRoot.position.set(-0.15, 0.56, 0.2);
sealRoot.rotation.y = -0.2;
world.add(sealRoot);

const sealRestPosition = new THREE.Vector3(-0.15, 0.56, 0.2);
const sealSwimCenter = new THREE.Vector3(3.45, -1.05, 4.6);
const sealSwimRight = new THREE.Vector3(0.8, 0, -0.6);

const sealMorph = new THREE.Group();
sealRoot.add(sealMorph);

const body = makeMesh(
  new THREE.SphereGeometry(1, 10, 7),
  colors.fur,
  [-0.35, 0.35, 0],
  [1.65, 0.72, 0.72],
  [0, 0, -0.08]
);
sealMorph.add(body);

const belly = makeMesh(
  new THREE.SphereGeometry(1, 9, 6),
  colors.furLight,
  [0.05, 0.12, 0.54],
  [0.85, 0.45, 0.18],
  [0, 0, -0.12]
);
sealMorph.add(belly);

const neck = makeMesh(
  new THREE.CylinderGeometry(0.54, 0.82, 1.12, 8),
  colors.fur,
  [0.85, 0.76, 0],
  [1, 1, 1],
  [0, 0, -0.72]
);
sealMorph.add(neck);

const headPivot = new THREE.Group();
headPivot.position.set(1.22, 1.02, 0);
sealMorph.add(headPivot);

const head = makeMesh(
  new THREE.SphereGeometry(0.64, 9, 7),
  colors.fur,
  [0, 0, 0],
  [1.02, 0.9, 0.92],
  [0, 0, -0.05]
);
headPivot.add(head);

const muzzle = makeMesh(
  new THREE.SphereGeometry(0.34, 8, 6),
  colors.furLight,
  [0.47, -0.12, 0.28],
  [1, 0.72, 0.75],
  [0, 0, 0]
);
headPivot.add(muzzle);

const nose = makeMesh(
  new THREE.SphereGeometry(0.13, 6, 4),
  colors.dark,
  [0.76, -0.06, 0.35],
  [1.1, 0.8, 0.8],
  [0, 0, 0]
);
headPivot.add(nose);

const mouthInterior = makeMesh(
  new THREE.SphereGeometry(0.22, 7, 5),
  colors.dark,
  [0.57, -0.24, 0.31],
  [1.05, 0.28, 0.7],
  [0, 0, 0]
);
headPivot.add(mouthInterior);

const jawPivot = new THREE.Group();
jawPivot.position.set(0.4, -0.2, 0.28);
headPivot.add(jawPivot);

const lowerJaw = makeMesh(
  new THREE.SphereGeometry(0.25, 7, 5),
  colors.furLight,
  [0.16, -0.08, 0.03],
  [1.05, 0.38, 0.72],
  [0, 0, 0]
);
jawPivot.add(lowerJaw);

const faceGroup = new THREE.Group();
faceGroup.rotation.set(-0.22, 0.68, -0.04);
headPivot.add(faceGroup);

const createEye = (x) => {
  const eye = new THREE.Group();
  eye.position.set(x, 0.17, 0.59);
  const white = makeMesh(
    new THREE.SphereGeometry(0.12, 7, 5),
    colors.eye,
    [0, 0, 0],
    [1, 1, 0.42],
    [0, 0, 0]
  );
  const pupil = makeMesh(
    new THREE.SphereGeometry(0.07, 6, 4),
    colors.dark,
    [0, -0.012, 0.055],
    [1, 1, 0.5],
    [0, 0, 0]
  );
  const glint = makeMesh(
    new THREE.SphereGeometry(0.02, 4, 3),
    colors.eye,
    [0.022, 0.025, 0.095],
    [1, 1, 0.5],
    [0, 0, 0]
  );
  eye.add(white, pupil, glint);
  faceGroup.add(eye);
  return { eye, glint };
};

const leftEyeParts = createEye(-0.145);
const rightEyeParts = createEye(0.145);
const leftEye = leftEyeParts.eye;
const farEye = rightEyeParts.eye;
const eyeGlint = leftEyeParts.glint;
const farEyeGlint = rightEyeParts.glint;

const ear = makeMesh(
  new THREE.SphereGeometry(0.12, 5, 4),
  colors.flipper,
  [-0.42, 0.16, 0.36],
  [0.45, 0.9, 0.35],
  [0, 0, 0.4]
);
headPivot.add(ear);

const makeWhisker = (start, end) => {
  const direction = new THREE.Vector3().subVectors(end, start);
  const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
  const whisker = makeMesh(
    new THREE.CylinderGeometry(0.008, 0.008, direction.length(), 4),
    colors.eye,
    [midpoint.x, midpoint.y, midpoint.z],
    [1, 1, 1],
    [0, 0, 0]
  );
  whisker.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.normalize()
  );
  whisker.castShadow = false;
  return whisker;
};

faceGroup.add(makeWhisker(new THREE.Vector3(-0.18, -0.13, 0.61), new THREE.Vector3(-0.82, -0.02, 0.64)));
faceGroup.add(makeWhisker(new THREE.Vector3(-0.2, -0.18, 0.61), new THREE.Vector3(-0.88, -0.2, 0.64)));
faceGroup.add(makeWhisker(new THREE.Vector3(-0.18, -0.23, 0.61), new THREE.Vector3(-0.8, -0.38, 0.64)));
faceGroup.add(makeWhisker(new THREE.Vector3(0.18, -0.13, 0.61), new THREE.Vector3(0.82, -0.02, 0.64)));
faceGroup.add(makeWhisker(new THREE.Vector3(0.2, -0.18, 0.61), new THREE.Vector3(0.88, -0.2, 0.64)));
faceGroup.add(makeWhisker(new THREE.Vector3(0.18, -0.23, 0.61), new THREE.Vector3(0.8, -0.38, 0.64)));

const frontFlipper = makeMesh(
  new THREE.CapsuleGeometry(0.2, 0.88, 3, 6),
  colors.flipper,
  [0.32, -0.04, 0.6],
  [1, 1, 0.55],
  [0.15, 0, 1.02]
);
sealMorph.add(frontFlipper);

const farFlipper = makeMesh(
  new THREE.CapsuleGeometry(0.18, 0.72, 3, 6),
  colors.flipper,
  [0.18, 0, -0.55],
  [1, 1, 0.5],
  [-0.2, 0, -0.92]
);
sealMorph.add(farFlipper);

const tailLeft = makeMesh(
  new THREE.CapsuleGeometry(0.25, 0.75, 3, 6),
  colors.flipper,
  [-1.75, 0.28, 0.19],
  [1, 1, 0.5],
  [0, 0.25, 1.15]
);
sealMorph.add(tailLeft);

const tailRight = makeMesh(
  new THREE.CapsuleGeometry(0.25, 0.75, 3, 6),
  colors.flipper,
  [-1.75, 0.28, -0.19],
  [1, 1, 0.5],
  [0, -0.25, 1.15]
);
sealMorph.add(tailRight);

const fishGroup = new THREE.Group();
fishGroup.visible = false;
world.add(fishGroup);
const fishStart = new THREE.Vector3(4.6, 1.55, 1.1);
const fishTarget = new THREE.Vector3();

const fishBody = makeMesh(
  new THREE.SphereGeometry(0.28, 7, 5),
  colors.fish,
  [0, 0, 0],
  [1.4, 0.72, 0.45],
  [0, 0, 0]
);
fishGroup.add(fishBody);

const fishTail = makeMesh(
  new THREE.ConeGeometry(0.32, 0.5, 3),
  colors.fishFin,
  [-0.58, 0, 0],
  [1, 1, 0.5],
  [0, 0, -Math.PI / 2]
);
fishGroup.add(fishTail);

const fishEye = makeMesh(
  new THREE.SphereGeometry(0.04, 4, 3),
  colors.dark,
  [0.26, 0.12, 0.13],
  [1, 1, 0.6],
  [0, 0, 0]
);
fishGroup.add(fishEye);

const swimmers = [];

const createSwimmer = (index) => {
  const swimmer = new THREE.Group();
  const swimmerMaterial = index % 2 ? colors.fur : colors.furLight;
  const swimmerBody = makeMesh(
    new THREE.SphereGeometry(0.52, 8, 6),
    swimmerMaterial,
    [0, 0, 0],
    [1.75, 0.58, 0.62],
    [0, 0, 0]
  );
  const swimmerNeck = makeMesh(
    new THREE.CylinderGeometry(0.3, 0.46, 0.72, 7),
    swimmerMaterial,
    [0.57, 0.22, 0],
    [1, 1, 1],
    [0, 0, -0.78]
  );
  const swimmerHead = makeMesh(
    new THREE.SphereGeometry(0.36, 8, 6),
    swimmerMaterial,
    [0.82, 0.38, 0],
    [1, 0.9, 0.9],
    [0, 0, 0]
  );
  const swimmerMuzzle = makeMesh(
    new THREE.SphereGeometry(0.16, 6, 4),
    colors.furLight,
    [1.1, 0.3, 0.15],
    [1, 0.7, 0.75],
    [0, 0, 0]
  );
  const swimmerNose = makeMesh(
    new THREE.SphereGeometry(0.07, 5, 3),
    colors.dark,
    [1.23, 0.31, 0.19],
    [1, 0.75, 0.65],
    [0, 0, 0]
  );
  const swimmerEyeLeft = makeMesh(
    new THREE.SphereGeometry(0.045, 5, 3),
    colors.dark,
    [0.94, 0.48, 0.27],
    [1, 1, 0.5],
    [0, 0, 0]
  );
  const swimmerEyeRight = makeMesh(
    new THREE.SphereGeometry(0.045, 5, 3),
    colors.dark,
    [1.08, 0.45, 0.23],
    [1, 1, 0.5],
    [0, 0, 0]
  );
  const swimmerFlipper = makeMesh(
    new THREE.CapsuleGeometry(0.13, 0.54, 2, 5),
    colors.flipper,
    [0.1, -0.18, 0.5],
    [1, 1, 0.65],
    [0.1, 0, 0.72]
  );
  const swimmerFarFlipper = makeMesh(
    new THREE.CapsuleGeometry(0.12, 0.48, 2, 5),
    colors.flipper,
    [0.08, -0.12, -0.48],
    [1, 1, 0.6],
    [-0.1, 0, -0.72]
  );
  const swimmerTailLeft = makeMesh(
    new THREE.CapsuleGeometry(0.15, 0.52, 2, 5),
    colors.flipper,
    [-1.02, 0, 0.18],
    [1, 1, 0.58],
    [0, 0.2, 1.16]
  );
  const swimmerTailRight = makeMesh(
    new THREE.CapsuleGeometry(0.15, 0.52, 2, 5),
    colors.flipper,
    [-1.02, 0, -0.18],
    [1, 1, 0.58],
    [0, -0.2, 1.16]
  );
  swimmer.add(
    swimmerBody,
    swimmerNeck,
    swimmerHead,
    swimmerMuzzle,
    swimmerNose,
    swimmerEyeLeft,
    swimmerEyeRight,
    swimmerFlipper,
    swimmerFarFlipper,
    swimmerTailLeft,
    swimmerTailRight
  );
  swimmer.userData.flipper = swimmerFlipper;
  swimmer.userData.farFlipper = swimmerFarFlipper;
  swimmer.userData.tailLeft = swimmerTailLeft;
  swimmer.userData.tailRight = swimmerTailRight;
  swimmer.scale.setScalar(0.82 + index * 0.07);
  swimmer.position.set(-8 + index * 4.8, -0.62, -2.8 - (index % 3) * 1.35);
  world.add(swimmer);
  swimmers.push(swimmer);
};

for (let index = 0; index < 4; index += 1) {
  createSwimmer(index);
}

const visitorSeal = swimmers[0].clone(true);
visitorSeal.visible = false;
visitorSeal.scale.setScalar(0.96);
world.add(visitorSeal);

const visitorStart = new THREE.Vector3(5.4, -0.72, 1.2);
const visitorRock = new THREE.Vector3(2.35, 0.22, 0.65);

const poopGroup = new THREE.Group();
poopGroup.visible = false;
world.add(poopGroup);

poopGroup.add(
  makeMesh(new THREE.SphereGeometry(0.22, 7, 5), colors.poop, [0, 0, 0], [1.2, 0.55, 1], [0, 0, 0]),
  makeMesh(new THREE.SphereGeometry(0.17, 7, 5), colors.poop, [0, 0.17, 0], [1, 0.65, 1], [0, 0, 0]),
  makeMesh(new THREE.ConeGeometry(0.12, 0.22, 6), colors.poop, [0, 0.34, 0], [1, 1, 1], [0, 0, 0])
);

for (let index = 0; index < 3; index += 1) {
  const wisp = makeMesh(
    new THREE.CapsuleGeometry(0.025, 0.3, 2, 4),
    colors.stink,
    [-0.18 + index * 0.18, 0.58, 0],
    [1, 1, 1],
    [0, 0, index % 2 ? -0.2 : 0.2]
  );
  wisp.userData.phase = index * 1.7;
  poopGroup.add(wisp);
}

const burpCloud = new THREE.Group();
burpCloud.visible = false;
world.add(burpCloud);

for (let index = 0; index < 5; index += 1) {
  const puff = makeMesh(
    new THREE.IcosahedronGeometry(0.18 + index * 0.025, 1),
    colors.stink,
    [index * 0.2, Math.sin(index) * 0.12, (index % 2) * 0.12],
    [1, 1, 1],
    [0, 0, 0]
  );
  puff.castShadow = false;
  puff.userData.phase = index * 0.9;
  burpCloud.add(puff);
}

const burpTarget = new THREE.Vector3();

const foamLines = [];

for (let lineIndex = 0; lineIndex < 5; lineIndex += 1) {
  const points = [];
  for (let pointIndex = 0; pointIndex < 16; pointIndex += 1) {
    points.push(new THREE.Vector3(-8 + pointIndex, -0.84, -3 - lineIndex * 2));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: "#c8f2df" }));
  line.userData.phase = lineIndex * 0.9;
  world.add(line);
  foamLines.push(line);
}

const clamp = (value) => Math.max(0, Math.min(100, value));

const updateBars = () => {
  Object.entries(ui).forEach(([key, item]) => {
    const value = Math.round(state[key]);
    item.value.textContent = String(value).padStart(2, "0");
    item.bar.style.width = `${value}%`;
    item.bar.style.backgroundColor = value < 25 ? "#ff654f" : "#55e0aa";
  });
};

const setSpeech = (message) => {
  speech.textContent = message;
};

let audioContext;

const createSealTone = (start, frequency, duration, volume) => {
  const oscillator = audioContext.createOscillator();
  const filter = audioContext.createBiquadFilter();
  const gain = audioContext.createGain();
  oscillator.type = "sawtooth";
  oscillator.frequency.setValueAtTime(frequency, start);
  oscillator.frequency.exponentialRampToValueAtTime(frequency * 0.55, start + duration * 0.58);
  oscillator.frequency.exponentialRampToValueAtTime(frequency * 0.82, start + duration);
  filter.type = "lowpass";
  filter.frequency.setValueAtTime(1250, start);
  filter.Q.setValueAtTime(4.5, start);
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(Math.min(volume * 2.4, 0.42), start + 0.025);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  oscillator.connect(filter);
  filter.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start(start);
  oscillator.stop(start + duration);
};

const playSealSound = (sound) => {
  if (!state.soundEnabled) return;
  audioContext ??= new AudioContext();
  if (audioContext.state === "suspended") audioContext.resume();
  const now = audioContext.currentTime + 0.015;
  if (sound === "bark") {
    createSealTone(now, 330, 0.2, 0.16);
    createSealTone(now + 0.24, 390, 0.18, 0.13);
  }
  if (sound === "honk") {
    createSealTone(now, 210, 0.46, 0.14);
  }
  if (sound === "snore") {
    createSealTone(now, 145, 0.58, 0.08);
    createSealTone(now + 0.66, 120, 0.52, 0.06);
  }
  if (sound === "achievement") {
    createSealTone(now, 260, 0.2, 0.15);
    createSealTone(now + 0.22, 360, 0.2, 0.15);
    createSealTone(now + 0.44, 510, 0.4, 0.17);
  }
  if (sound === "song") {
    [430, 430, 270, 430, 270, 270].forEach((frequency, index) => {
      createSealTone(now + index * 0.19, frequency, 0.17, 0.12);
    });
  }
  if (sound === "burp") {
    createSealTone(now, 125, 0.72, 0.18);
    createSealTone(now + 0.12, 92, 0.78, 0.14);
  }
};

const updateSoundToggle = () => {
  soundToggle.textContent = state.soundEnabled ? "SOUND ON" : "SOUND OFF";
  soundToggle.setAttribute("aria-pressed", String(state.soundEnabled));
};

const toggleSound = () => {
  state.soundEnabled = !state.soundEnabled;
  localStorage.setItem("seagotchi-sound", state.soundEnabled ? "on" : "off");
  updateSoundToggle();
  setSpeech(state.soundEnabled ? "AE AE AU AE AU AU!" : "QUIET COVE MODE.");
  if (state.soundEnabled) playSealSound("song");
};

const updateDay = () => {
  dayNumber.textContent = String(state.day).padStart(2, "0");
};

const startAmbientEvent = (seconds) => {
  const events = ["climb", "poop", "burp"];
  state.ambientEvent = events[state.ambientIndex % events.length];
  state.ambientIndex += 1;
  state.ambientStarted = seconds;
  if (state.ambientEvent === "climb") {
    visitorSeal.visible = true;
    visitorSeal.position.copy(visitorStart);
    visitorSeal.rotation.set(0, 0, 0);
    setSpeech("HEY! THIS IS MY ROCK!");
    playSealSound("bark");
  }
  if (state.ambientEvent === "poop") {
    poopGroup.visible = true;
    setSpeech("UH-OH... THE TIDE WILL GET THAT.");
    playSealSound("honk");
  }
  if (state.ambientEvent === "burp") {
    burpCloud.visible = true;
    setSpeech("BUUUURP! THAT WAS A STINKY ONE.");
    playSealSound("burp");
  }
};

const finishAmbientEvent = (seconds) => {
  visitorSeal.visible = false;
  poopGroup.visible = false;
  burpCloud.visible = false;
  state.ambientEvent = null;
  state.nextAmbientAt = seconds + 20 + Math.random() * 20;
  setSpeech("BORK! THE TIDE IS NICE.");
};

const updateSize = () => {
  const growth = Math.min(state.feeds, 12);
  sealMorph.scale.set(
    1 + growth * 0.055,
    1 + growth * 0.042,
    1 + growth * 0.065
  );
  if (growth === 0) weightValue.textContent = "SLEEK";
  if (growth >= 1) weightValue.textContent = "FAT-FAT";
  if (growth >= 3) weightValue.textContent = "SUPER-FAT";
  if (growth >= 5) weightValue.textContent = "UBER-FAT";
  if (growth >= 8) weightValue.textContent = "CHONKERS";
};

const showAchievement = () => {
  if (state.achievement) return;
  state.achievement = true;
  achievement.classList.add("visible");
  createConfetti();
  setSpeech("BORK! I HAVE BECOME UNSTOPPABLE.");
  playSealSound("achievement");
  window.setTimeout(() => achievement.classList.remove("visible"), 4200);
};

const createConfetti = () => {
  const colors = ["#ffd15c", "#ff654f", "#55e0aa", "#d9f7e8", "#1496c4"];
  confettiLayer.replaceChildren();
  for (let index = 0; index < 72; index += 1) {
    const confetti = document.createElement("span");
    const sway = Math.round(Math.random() * 160 - 80);
    const turn = Math.round(Math.random() * 720 + 360);
    const size = Math.round(Math.random() * 5 + 5);
    confetti.className = "confetti";
    confetti.style.setProperty("--x", `${Math.random() * 100}%`);
    confetti.style.setProperty("--sway", `${sway}px`);
    confetti.style.setProperty("--sway-back", `${Math.round(sway * -0.55)}px`);
    confetti.style.setProperty("--turn", `${turn}deg`);
    confetti.style.setProperty("--turn-mid", `${Math.round(turn * 0.4)}deg`);
    confetti.style.setProperty("--turn-late", `${Math.round(turn * 0.72)}deg`);
    confetti.style.setProperty("--delay", `${Math.random() * 0.55}s`);
    confetti.style.setProperty("--duration", `${Math.random() * 1.2 + 2.6}s`);
    confetti.style.setProperty("--size", `${size}px`);
    confetti.style.setProperty("--height", `${Math.round(size * 1.7)}px`);
    confetti.style.setProperty("--color", colors[index % colors.length]);
    confettiLayer.append(confetti);
  }
  window.setTimeout(() => confettiLayer.replaceChildren(), 4400);
};

const feed = () => {
  if (state.action === "feed" || state.action === "swim") return;
  if (state.sleeping) {
    state.sleeping = false;
    sleepLabel.textContent = "PUT TO SLEEP";
  }
  state.action = "feed";
  state.actionStarted = performance.now();
  feedButton.disabled = true;
  swimButton.disabled = true;
  fishGroup.visible = true;
  fishGroup.scale.setScalar(1);
  fishGroup.rotation.set(0, Math.PI, 0);
  state.food = clamp(state.food + 18);
  state.happy = clamp(state.happy + 5);
  setSpeech("FISH! FISH! FISH!");
  playSealSound("bark");
  updateBars();
  window.setTimeout(() => {
    state.feeds += 1;
    updateSize();
    fishGroup.visible = false;
    fishGroup.scale.setScalar(1);
    feedButton.disabled = false;
    swimButton.disabled = false;
    state.action = "idle";
    setSpeech(state.feeds >= 6 ? "ONE MORE COULD NOT HURT..." : "TASTES LIKE THE PACIFIC.");
    if (state.feeds >= 8) showAchievement();
  }, 1450);
};

const toggleSleep = () => {
  if (state.action === "swim") return;
  state.sleeping = !state.sleeping;
  state.action = state.sleeping ? "sleep" : "idle";
  state.actionStarted = performance.now();
  sleepLabel.textContent = state.sleeping ? "WAKE UP" : "PUT TO SLEEP";
  setSpeech(state.sleeping ? "HONK... SHOOO... HONK..." : "THE SUN IS BACK!");
  playSealSound(state.sleeping ? "snore" : "honk");
};

const createHearts = () => {
  for (let index = 0; index < 5; index += 1) {
    const heart = document.createElement("span");
    heart.className = "heart";
    heart.textContent = "♥";
    heart.style.setProperty("--drift", `${(index - 2) * 42}px`);
    heart.style.animationDelay = `${index * 0.09}s`;
    heartLayer.append(heart);
    window.setTimeout(() => heart.remove(), 1700);
  }
};

const pet = () => {
  if (state.action === "swim") return;
  state.action = "pet";
  state.actionStarted = performance.now();
  state.happy = clamp(state.happy + 15);
  setSpeech("BORK BORK! DO THAT AGAIN.");
  playSealSound("song");
  createHearts();
  updateBars();
  window.setTimeout(() => {
    if (state.action === "pet") state.action = "idle";
  }, 1300);
};

const swim = () => {
  if (state.action === "feed" || state.action === "swim") return;
  if (state.ambientEvent) finishAmbientEvent(performance.now() / 1000);
  state.sleeping = false;
  sleepLabel.textContent = "PUT TO SLEEP";
  state.action = "swim";
  state.actionStarted = performance.now();
  state.swimDuration = 3000 + Math.random() * 2000;
  feedButton.disabled = true;
  sleepButton.disabled = true;
  petButton.disabled = true;
  swimButton.disabled = true;
  swimButton.setAttribute("aria-busy", "true");
  swimLabel.textContent = "SWIMMING";
  setSpeech("SPLASH! COVE LAP TIME.");
  playSealSound("honk");
  window.setTimeout(() => {
    if (state.action !== "swim") return;
    state.action = "idle";
    sealRoot.position.copy(sealRestPosition);
    sealRoot.rotation.set(0, -0.2, 0);
    sealRoot.scale.setScalar(1);
    frontFlipper.rotation.z = 1.02;
    farFlipper.rotation.z = -0.92;
    tailLeft.rotation.z = 1.15;
    tailRight.rotation.z = 1.15;
    feedButton.disabled = false;
    sleepButton.disabled = false;
    petButton.disabled = false;
    swimButton.disabled = false;
    swimButton.removeAttribute("aria-busy");
    swimLabel.textContent = "GO SWIM";
    setSpeech("BEST LAP YET. BORK!");
    playSealSound("bark");
  }, state.swimDuration);
};

feedButton.addEventListener("click", feed);
sleepButton.addEventListener("click", toggleSleep);
petButton.addEventListener("click", pet);
swimButton.addEventListener("click", swim);
soundToggle.addEventListener("click", toggleSound);

canvas.addEventListener("click", pet);

let pointerDown = false;
let pointerX = 0;
let targetRotation = 0;

canvas.addEventListener("pointerdown", (event) => {
  pointerDown = true;
  pointerX = event.clientX;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (!pointerDown) return;
  const movement = event.clientX - pointerX;
  pointerX = event.clientX;
  targetRotation += movement * 0.006;
});

canvas.addEventListener("pointerup", () => {
  pointerDown = false;
});

const updateClock = () => {
  const now = new Date();
  document.querySelector("#clock").textContent = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit"
  });
};

const resize = () => {
  const width = sceneWrap.clientWidth;
  const height = sceneWrap.clientHeight;
  const aspect = width / height;
  const viewHeight = 9;
  camera.left = (-viewHeight * aspect) / 2;
  camera.right = (viewHeight * aspect) / 2;
  camera.top = viewHeight / 2;
  camera.bottom = -viewHeight / 2;
  camera.updateProjectionMatrix();
  renderer.setSize(Math.max(1, Math.floor(width * 0.62)), Math.max(1, Math.floor(height * 0.62)), false);
};

const syncViewport = () => {
  const viewportHeight = window.visualViewport?.height ?? window.innerHeight;
  const mobileDevice = /Android|iPad|iPhone|iPod/i.test(navigator.userAgent) ||
    window.matchMedia("(pointer: coarse)").matches;
  const mobileLayout = mobileDevice &&
    (window.innerWidth <= 1024 || window.innerHeight <= 600);
  document.documentElement.style.setProperty("--app-height", `${Math.round(viewportHeight)}px`);
  document.body.classList.toggle("mobile-layout", mobileLayout);
  resize();
};

new ResizeObserver(resize).observe(sceneWrap);
window.addEventListener("resize", syncViewport);
window.addEventListener("orientationchange", syncViewport);
window.visualViewport?.addEventListener("resize", syncViewport);

let previousTime = performance.now();

const animate = (time) => {
  const delta = Math.min((time - previousTime) / 1000, 0.1);
  previousTime = time;
  const seconds = time / 1000;
  state.elapsedCare += delta;
  state.elapsedDay += delta;

  oceanMaterial.uniforms.time.value = seconds;
  world.rotation.y += (targetRotation - world.rotation.y) * 0.045;

  foamLines.forEach((line, lineIndex) => {
    const positions = line.geometry.attributes.position;
    for (let index = 0; index < positions.count; index += 1) {
      positions.setY(index, -0.84 + Math.sin(index * 0.65 + seconds * 1.9 + line.userData.phase) * 0.08);
    }
    positions.needsUpdate = true;
    line.position.x = Math.sin(seconds * 0.35 + lineIndex) * 0.8;
  });

  swimmers.forEach((swimmer, index) => {
    swimmer.position.x += delta * (0.5 + index * 0.11);
    if (swimmer.position.x > 9) swimmer.position.x = -10;
    swimmer.position.y = -0.72 + Math.sin(seconds * 1.5 + index * 1.7) * 0.22;
    swimmer.rotation.z = Math.sin(seconds * 2.1 + index) * 0.08;
    const stroke = Math.sin(seconds * 3.2 + index * 1.3);
    swimmer.userData.flipper.rotation.z = 0.72 + stroke * 0.38;
    swimmer.userData.farFlipper.rotation.z = -0.72 - stroke * 0.3;
    swimmer.userData.tailLeft.rotation.z = 1.16 + stroke * 0.2;
    swimmer.userData.tailRight.rotation.z = 1.16 - stroke * 0.2;
  });

  const breathing = Math.sin(seconds * 2.2) * 0.025;
  body.scale.y = 0.72 + breathing;
  sealRoot.position.y = 0.56 + Math.sin(seconds * 1.5) * 0.015;

  if (state.action === "swim") {
    const elapsed = time - state.actionStarted;
    const progress = Math.min(elapsed / state.swimDuration, 1);
    const swimRadius = Math.min(2.8, (camera.right - camera.left) * 0.28);
    const swimScale = Math.min(0.76, 0.55 + swimRadius * 0.06);
    const swimHeading = -0.2 - Math.atan2(sealSwimRight.z, sealSwimRight.x);
    if (progress < 0.18) {
      const stage = progress / 0.18;
      const ease = stage * stage * (3 - 2 * stage);
      sealRoot.position.set(
        sealRestPosition.x + (sealSwimCenter.x - sealRestPosition.x) * ease,
        sealRestPosition.y + (sealSwimCenter.y - sealRestPosition.y) * ease + Math.sin(stage * Math.PI) * 1.35,
        sealRestPosition.z + (sealSwimCenter.z - sealRestPosition.z) * ease
      );
      sealRoot.rotation.set(0, -0.2 + (swimHeading + 0.2) * ease, -Math.sin(stage * Math.PI) * 0.34);
      sealRoot.scale.setScalar(1 + (swimScale - 1) * ease);
    } else if (progress < 0.8) {
      const stage = (progress - 0.18) / 0.62;
      const angle = stage * Math.PI * 2;
      const offset = swimRadius * Math.sin(angle);
      const direction = swimRadius * Math.cos(angle);
      const directionX = sealSwimRight.x * direction;
      const directionZ = sealSwimRight.z * direction;
      sealRoot.position.set(
        sealSwimCenter.x + sealSwimRight.x * offset,
        sealSwimCenter.y + Math.sin(angle * 2) * 0.08,
        sealSwimCenter.z + sealSwimRight.z * offset
      );
      sealRoot.rotation.set(
        0,
        -0.2 - Math.atan2(directionZ, directionX),
        Math.sin(angle * 2) * 0.08
      );
      sealRoot.scale.setScalar(swimScale);
    } else {
      const stage = (progress - 0.8) / 0.2;
      const ease = stage * stage * (3 - 2 * stage);
      sealRoot.position.set(
        sealSwimCenter.x + (sealRestPosition.x - sealSwimCenter.x) * ease,
        sealSwimCenter.y + (sealRestPosition.y - sealSwimCenter.y) * ease + Math.sin(stage * Math.PI) * 1.35,
        sealSwimCenter.z + (sealRestPosition.z - sealSwimCenter.z) * ease
      );
      sealRoot.rotation.set(0, swimHeading + (-0.2 - swimHeading) * ease, -Math.sin(stage * Math.PI) * 0.38);
      sealRoot.scale.setScalar(swimScale + ease * (1 - swimScale));
    }
    const stroke = Math.sin(elapsed * 0.018);
    headPivot.rotation.z = Math.sin(elapsed * 0.012) * 0.1;
    frontFlipper.rotation.z = 1.02 + stroke * 0.72;
    farFlipper.rotation.z = -0.92 - stroke * 0.62;
    tailLeft.rotation.z = 1.15 + stroke * 0.38;
    tailRight.rotation.z = 1.15 - stroke * 0.38;
  } else if (state.action === "feed") {
    const progress = Math.min((time - state.actionStarted) / 1350, 1);
    const approach = progress * progress * (3 - 2 * progress);
    const mouthRise = Math.sin(Math.min(progress / 0.35, 1) * Math.PI * 0.5);
    const mouthClose = progress < 0.9 ? 1 : Math.max(0, 1 - (progress - 0.9) / 0.1);
    const mouthOpen = mouthRise * mouthClose;
    const eaten = Math.max(0, (progress - 0.88) / 0.12);
    headPivot.rotation.z = Math.sin(progress * Math.PI) * 0.08;
    jawPivot.rotation.z = -mouthOpen * 0.48;
    mouthInterior.scale.y = 0.28 + mouthOpen * 1.35;
    world.updateMatrixWorld(true);
    mouthInterior.getWorldPosition(fishTarget);
    world.worldToLocal(fishTarget);
    fishGroup.position.lerpVectors(fishStart, fishTarget, approach);
    fishGroup.position.y += Math.sin(progress * Math.PI) * 0.35;
    fishGroup.rotation.z = Math.sin(progress * 28) * 0.18;
    fishGroup.scale.setScalar(Math.max(0.02, 1 - eaten));
  } else if (state.action === "pet") {
    headPivot.rotation.z = Math.sin((time - state.actionStarted) * 0.024) * 0.25;
    frontFlipper.rotation.z = 1.02 + Math.sin((time - state.actionStarted) * 0.03) * 0.5;
  } else if (state.sleeping) {
    headPivot.rotation.z += (-0.42 - headPivot.rotation.z) * 0.06;
    headPivot.position.y = 1.02 + Math.sin(seconds * 1.2) * 0.03;
    leftEye.scale.y = 0.12;
    farEye.scale.y = 0.12;
    eyeGlint.visible = false;
    farEyeGlint.visible = false;
  } else {
    headPivot.rotation.z *= 0.9;
    headPivot.position.y += (1.02 - headPivot.position.y) * 0.08;
    frontFlipper.rotation.z += (1.02 - frontFlipper.rotation.z) * 0.08;
    leftEye.scale.y += (1 - leftEye.scale.y) * 0.14;
    farEye.scale.y += (1 - farEye.scale.y) * 0.14;
    eyeGlint.visible = true;
    farEyeGlint.visible = true;
  }

  if (state.action !== "feed") {
    jawPivot.rotation.z *= 0.82;
    mouthInterior.scale.y += (0.28 - mouthInterior.scale.y) * 0.18;
  }

  if (
    !state.ambientEvent &&
    seconds >= state.nextAmbientAt &&
    state.action === "idle" &&
    !state.sleeping
  ) {
    startAmbientEvent(seconds);
  }

  if (state.ambientEvent) {
    const ambientElapsed = seconds - state.ambientStarted;
    let ambientDuration = 4;

    if (state.ambientEvent === "climb") {
      ambientDuration = 4.2;
      if (ambientElapsed < 2.2) {
        const climbProgress = Math.min(ambientElapsed / 2.2, 1);
        const climbEase = climbProgress * climbProgress * (3 - 2 * climbProgress);
        visitorSeal.position.lerpVectors(visitorStart, visitorRock, climbEase);
        visitorSeal.rotation.z = Math.sin(climbProgress * Math.PI) * 0.12;
      } else {
        const kickProgress = Math.min((ambientElapsed - 2.2) / 2, 1);
        visitorSeal.position.set(
          visitorRock.x + kickProgress * 4.8,
          visitorRock.y + Math.sin(kickProgress * Math.PI) * 1.1 - kickProgress * 1.25,
          visitorRock.z + kickProgress * 0.7
        );
        visitorSeal.rotation.z = -kickProgress * 2.8;
        frontFlipper.rotation.z = 1.02 + Math.sin(kickProgress * Math.PI) * 1.65;
      }
    }

    if (state.ambientEvent === "poop") {
      ambientDuration = 6;
      const dropProgress = Math.min(ambientElapsed / 0.85, 1);
      poopGroup.position.set(-1.35, 1.28 - dropProgress * 0.58, -0.28);
      tailLeft.rotation.z = 1.15 + Math.sin(ambientElapsed * 9) * 0.22;
      tailRight.rotation.z = 1.15 - Math.sin(ambientElapsed * 9) * 0.22;
      poopGroup.children.slice(3).forEach((wisp, index) => {
        wisp.position.y = 0.54 + Math.sin(ambientElapsed * 2.4 + wisp.userData.phase) * 0.12;
        wisp.scale.y = 0.8 + Math.sin(ambientElapsed * 2 + index) * 0.22;
      });
    }

    if (state.ambientEvent === "burp") {
      ambientDuration = 3.4;
      const burpProgress = Math.min(ambientElapsed / ambientDuration, 1);
      world.updateMatrixWorld(true);
      mouthInterior.getWorldPosition(burpTarget);
      world.worldToLocal(burpTarget);
      burpCloud.position.copy(burpTarget);
      burpCloud.position.x += burpProgress * 1.45;
      burpCloud.position.y += Math.sin(burpProgress * Math.PI) * 0.45;
      burpCloud.scale.setScalar(0.35 + burpProgress * 1.25);
      burpCloud.rotation.z = Math.sin(ambientElapsed * 3) * 0.15;
      if (state.action !== "feed") {
        const burpOpen = Math.sin(Math.min(ambientElapsed / 1.25, 1) * Math.PI);
        jawPivot.rotation.z = -burpOpen * 0.45;
        mouthInterior.scale.y = 0.28 + burpOpen * 1.25;
      }
    }

    if (ambientElapsed >= ambientDuration) finishAmbientEvent(seconds);
  }

  if (state.elapsedCare >= 1) {
    state.elapsedCare = 0;
    state.food = clamp(state.food - 0.18);
    if (state.sleeping) {
      state.sleep = clamp(state.sleep + 0.9);
      state.happy = clamp(state.happy + 0.08);
    } else {
      state.sleep = clamp(state.sleep - 0.1);
      state.happy = clamp(state.happy - (state.food < 25 ? 0.22 : 0.08));
    }
    if (state.food < 20 && !state.sleeping) setSpeech("MY TUMMY SOUNDS LIKE A STORM...");
    if (state.sleep < 18 && !state.sleeping) setSpeech("I NEED A ROCK NAP.");
    updateBars();
  }

  if (state.elapsedDay >= 60) {
    state.elapsedDay -= 60;
    state.day += 1;
    localStorage.setItem("seagotchi-day", String(state.day));
    updateDay();
    setSpeech("NEW DAY! AE AE AU AE AU AU!");
    playSealSound("song");
  }

  renderer.render(scene, camera);
  requestAnimationFrame(animate);
};

updateBars();
updateSize();
updateClock();
updateDay();
updateSoundToggle();
window.setInterval(updateClock, 30000);
syncViewport();
requestAnimationFrame(animate);
