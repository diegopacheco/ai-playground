import * as THREE from "three";
import "./style.css";

const canvas = document.querySelector("#ocean-canvas");
const sceneWrap = document.querySelector("#scene-wrap");
const feedButton = document.querySelector("#feed-button");
const sleepButton = document.querySelector("#sleep-button");
const petButton = document.querySelector("#pet-button");
const sleepLabel = document.querySelector("#sleep-label");
const speech = document.querySelector("#speech");
const achievement = document.querySelector("#achievement");
const weightValue = document.querySelector("#weight-value");
const heartLayer = document.querySelector("#heart-layer");

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
  achievement: false,
  elapsedCare: 0
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
  foam: new THREE.MeshBasicMaterial({ color: "#e2fff3" })
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

const leftEye = makeMesh(
  new THREE.SphereGeometry(0.09, 6, 4),
  colors.dark,
  [0.24, 0.16, 0.48],
  [1, 1, 0.5],
  [0, 0, 0]
);
headPivot.add(leftEye);

const eyeGlint = makeMesh(
  new THREE.SphereGeometry(0.025, 4, 3),
  colors.eye,
  [0.275, 0.2, 0.525],
  [1, 1, 0.4],
  [0, 0, 0]
);
headPivot.add(eyeGlint);

const farEye = makeMesh(
  new THREE.SphereGeometry(0.075, 6, 4),
  colors.dark,
  [0.35, 0.17, -0.32],
  [1, 1, 0.5],
  [0, 0, 0]
);
headPivot.add(farEye);

const ear = makeMesh(
  new THREE.SphereGeometry(0.12, 5, 4),
  colors.flipper,
  [-0.42, 0.16, 0.36],
  [0.45, 0.9, 0.35],
  [0, 0, 0.4]
);
headPivot.add(ear);

const makeWhisker = (y, angle) => {
  const whisker = makeMesh(
    new THREE.CylinderGeometry(0.008, 0.008, 0.72, 4),
    colors.eye,
    [0.78, y, 0.47],
    [1, 1, 1],
    [0, 0, angle]
  );
  whisker.rotation.x = Math.PI / 2.8;
  return whisker;
};

headPivot.add(makeWhisker(-0.12, -0.85));
headPivot.add(makeWhisker(-0.2, -1.03));
headPivot.add(makeWhisker(-0.28, -1.22));

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
  const swimmerBody = makeMesh(
    new THREE.SphereGeometry(0.5, 7, 5),
    index % 2 ? colors.fur : colors.furLight,
    [0, 0, 0],
    [1.7, 0.52, 0.5],
    [0, 0, 0]
  );
  const swimmerHead = makeMesh(
    new THREE.SphereGeometry(0.35, 7, 5),
    index % 2 ? colors.fur : colors.furLight,
    [0.72, 0.25, 0],
    [1, 1, 0.9],
    [0, 0, 0]
  );
  const swimmerNose = makeMesh(
    new THREE.SphereGeometry(0.08, 5, 3),
    colors.dark,
    [1.04, 0.24, 0.08],
    [1, 0.75, 0.65],
    [0, 0, 0]
  );
  swimmer.add(swimmerBody, swimmerHead, swimmerNose);
  swimmer.scale.setScalar(0.75 + index * 0.08);
  swimmer.position.set(-8 + index * 5, -0.72, -4 - index);
  world.add(swimmer);
  swimmers.push(swimmer);
};

for (let index = 0; index < 4; index += 1) {
  createSwimmer(index);
}

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

const updateSize = () => {
  const growth = Math.min(state.feeds, 12);
  sealMorph.scale.set(
    1 + growth * 0.055,
    1 + growth * 0.042,
    1 + growth * 0.065
  );
  if (growth < 3) weightValue.textContent = "SLEEK";
  if (growth >= 3) weightValue.textContent = "PLUSH";
  if (growth >= 6) weightValue.textContent = "HEFTY";
  if (growth >= 8) weightValue.textContent = "CHONKERS";
};

const showAchievement = () => {
  if (state.achievement) return;
  state.achievement = true;
  achievement.classList.add("visible");
  setSpeech("BORK! I HAVE BECOME UNSTOPPABLE.");
  window.setTimeout(() => achievement.classList.remove("visible"), 4200);
};

const feed = () => {
  if (state.action === "feed") return;
  if (state.sleeping) {
    state.sleeping = false;
    sleepLabel.textContent = "PUT TO SLEEP";
  }
  state.action = "feed";
  state.actionStarted = performance.now();
  feedButton.disabled = true;
  fishGroup.visible = true;
  state.food = clamp(state.food + 18);
  state.happy = clamp(state.happy + 5);
  setSpeech("FISH! FISH! FISH!");
  updateBars();
  window.setTimeout(() => {
    state.feeds += 1;
    updateSize();
    fishGroup.visible = false;
    feedButton.disabled = false;
    state.action = "idle";
    setSpeech(state.feeds >= 6 ? "ONE MORE COULD NOT HURT..." : "TASTES LIKE THE PACIFIC.");
    if (state.feeds >= 8) showAchievement();
  }, 1450);
};

const toggleSleep = () => {
  state.sleeping = !state.sleeping;
  state.action = state.sleeping ? "sleep" : "idle";
  state.actionStarted = performance.now();
  sleepLabel.textContent = state.sleeping ? "WAKE UP" : "PUT TO SLEEP";
  setSpeech(state.sleeping ? "HONK... SHOOO... HONK..." : "THE SUN IS BACK!");
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
  state.action = "pet";
  state.actionStarted = performance.now();
  state.happy = clamp(state.happy + 15);
  setSpeech("BORK BORK! DO THAT AGAIN.");
  createHearts();
  updateBars();
  window.setTimeout(() => {
    if (state.action === "pet") state.action = "idle";
  }, 1300);
};

feedButton.addEventListener("click", feed);
sleepButton.addEventListener("click", toggleSleep);
petButton.addEventListener("click", pet);

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
  const started = new Date(now.getFullYear(), 0, 1);
  const day = Math.floor((now - started) / 86400000) + 1;
  document.querySelector("#day-number").textContent = String(day).slice(-2).padStart(2, "0");
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

new ResizeObserver(resize).observe(sceneWrap);

let previousTime = performance.now();

const animate = (time) => {
  const delta = Math.min((time - previousTime) / 1000, 0.1);
  previousTime = time;
  const seconds = time / 1000;
  state.elapsedCare += delta;

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
  });

  const breathing = Math.sin(seconds * 2.2) * 0.025;
  body.scale.y = 0.72 + breathing;
  sealRoot.position.y = 0.56 + Math.sin(seconds * 1.5) * 0.015;

  if (state.action === "feed") {
    const progress = Math.min((time - state.actionStarted) / 1350, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    fishGroup.position.set(
      4.3 + (1.63 - 4.3) * eased,
      2.2 + (1.15 - 2.2) * eased + Math.sin(progress * Math.PI) * 1.2,
      1.5 + (0.62 - 1.5) * eased
    );
    fishGroup.rotation.z = Math.sin(progress * 28) * 0.18;
    headPivot.rotation.z = Math.sin(progress * 16) * 0.12;
  } else if (state.action === "pet") {
    headPivot.rotation.z = Math.sin((time - state.actionStarted) * 0.024) * 0.25;
    frontFlipper.rotation.z = 1.02 + Math.sin((time - state.actionStarted) * 0.03) * 0.5;
  } else if (state.sleeping) {
    headPivot.rotation.z += (-0.42 - headPivot.rotation.z) * 0.06;
    headPivot.position.y = 1.02 + Math.sin(seconds * 1.2) * 0.03;
    leftEye.scale.y = 0.12;
    farEye.scale.y = 0.12;
    eyeGlint.visible = false;
  } else {
    headPivot.rotation.z *= 0.9;
    headPivot.position.y += (1.02 - headPivot.position.y) * 0.08;
    frontFlipper.rotation.z += (1.02 - frontFlipper.rotation.z) * 0.08;
    leftEye.scale.y += (1 - leftEye.scale.y) * 0.14;
    farEye.scale.y += (1 - farEye.scale.y) * 0.14;
    eyeGlint.visible = true;
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

  renderer.render(scene, camera);
  requestAnimationFrame(animate);
};

updateBars();
updateSize();
updateClock();
window.setInterval(updateClock, 30000);
resize();
requestAnimationFrame(animate);
