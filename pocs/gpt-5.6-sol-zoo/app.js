const habitats = [
  {
    title: "Polar<br>Frontier",
    region: "UNITED STATES",
    description: "Where frozen coast, black water and white silence meet.",
    species: ["Penguins", "Polar bear", "Sea lion"],
    image: "assets/arctic.png",
    accent: "#dff5ff",
    weather: "snow",
    condition: "−18°C · HEAVY SNOW",
    coordinates: "71.2906° N · 156.7886° W",
    animals: [[18, 55], [68, 54], [85, 72]]
  },
  {
    title: "Emerald<br>Pulse",
    region: "BRAZIL",
    description: "Rain gathers on every leaf. The forest watches back.",
    species: ["Tree frog", "Jaguar", "Coral snake", "Monkeys"],
    image: "assets/brazil.png",
    accent: "#bff772",
    weather: "rain",
    condition: "29°C · TROPICAL RAIN",
    coordinates: "3.4653° S · 62.2159° W",
    animals: [[18, 77], [64, 56], [84, 52], [28, 24]]
  },
  {
    title: "Desert<br>Wind",
    region: "EGYPT",
    description: "Across the open sand, every track becomes a story.",
    species: ["Camel", "Arabian horse", "Scorpion", "Vulture"],
    image: "assets/egypt.png",
    accent: "#ffd277",
    weather: "sand",
    condition: "38°C · STRONG WIND",
    coordinates: "25.6872° N · 32.6396° E",
    animals: [[72, 48], [48, 58], [85, 84], [35, 16]]
  },
  {
    title: "Blue<br>Current",
    region: "ASIA PACIFIC",
    description: "An entire world moves on the pull of an unseen tide.",
    species: ["Reef fish", "Whale", "Shark", "Coral reef"],
    image: "assets/asia.png",
    accent: "#62f4ff",
    weather: "current",
    condition: "24°C · STRONG CURRENT",
    coordinates: "7.5149° N · 134.5825° E",
    animals: [[38, 36], [61, 46], [77, 28]]
  },
  {
    title: "Golden<br>Kingdom",
    region: "AFRICA",
    description: "The last light crosses the grass, and the plain comes alive.",
    species: ["Tiger", "Lion", "Elephant", "Meerkat", "Boar", "Cheetah"],
    image: "assets/africa.png",
    accent: "#ffbf69",
    weather: "dust",
    condition: "31°C · DRY DUST",
    coordinates: "2.3333° S · 34.8333° E",
    animals: [[14, 69], [48, 57], [65, 45], [58, 79], [84, 77], [88, 57]]
  }
];

const scenes = document.querySelector("#scenes");
const sceneNav = document.querySelector("#sceneNav");
const story = document.querySelector(".story");
const weather = document.querySelector("#weather");
const context = weather.getContext("2d");
const soundButton = document.querySelector("#sound");
const reaction = document.querySelector("#reaction");
const toast = document.querySelector("#toast");
let active = 0;
let locked = false;
let muted = false;
let audio;
let noiseSource;
let gain;
let particles = [];
let toastTimer;

habitats.forEach((habitat, index) => {
  const scene = document.createElement("div");
  scene.className = `scene${index === 0 ? " active" : ""}`;
  scene.style.setProperty("--image", `url(${habitat.image})`);
  const image = document.createElement("div");
  image.className = "scene-image";
  scene.append(image);
  habitat.animals.forEach(([x, y], animalIndex) => {
    const motion = document.createElement("div");
    motion.className = "animal-motion";
    motion.style.setProperty("--x", `${x}%`);
    motion.style.setProperty("--y", `${y}%`);
    motion.style.setProperty("--speed", `${2.8 + animalIndex * .55}s`);
    motion.style.setProperty("--delay", `${animalIndex * -.7}s`);
    scene.append(motion);
  });
  scenes.append(scene);
  const button = document.createElement("button");
  button.type = "button";
  button.className = index === 0 ? "active" : "";
  button.dataset.number = String(index + 1).padStart(2, "0");
  button.setAttribute("aria-label", `Open ${habitat.region} habitat`);
  button.addEventListener("click", event => {
    event.stopPropagation();
    showScene(index);
  });
  sceneNav.append(button);
});

function showScene(index) {
  if (index === active || locked) return;
  locked = true;
  active = (index + habitats.length) % habitats.length;
  const habitat = habitats[active];
  document.querySelectorAll(".scene").forEach((scene, sceneIndex) => scene.classList.toggle("active", sceneIndex === active));
  document.querySelectorAll(".scene-nav button").forEach((button, buttonIndex) => button.classList.toggle("active", buttonIndex === active));
  document.documentElement.style.setProperty("--accent", habitat.accent);
  story.classList.remove("changing");
  void story.offsetWidth;
  document.querySelector("#sceneNumber").textContent = String(active + 1).padStart(2, "0");
  document.querySelector("#region").textContent = habitat.region;
  document.querySelector("#title").innerHTML = habitat.title;
  document.querySelector("#description").textContent = habitat.description;
  document.querySelector("#species").innerHTML = habitat.species.map(name => `<span>${name}</span>`).join("");
  document.querySelector("#coordinates").textContent = habitat.coordinates;
  document.querySelector("#condition").textContent = habitat.condition;
  story.classList.add("changing");
  createParticles();
  tuneAudio();
  setTimeout(() => locked = false, 900);
}

function navigate(direction) {
  showScene((active + direction + habitats.length) % habitats.length);
}

window.addEventListener("wheel", event => {
  if (Math.abs(event.deltaY) < 8) return;
  navigate(event.deltaY > 0 ? 1 : -1);
}, { passive: true });

let touchY = 0;
window.addEventListener("touchstart", event => touchY = event.touches[0].clientY, { passive: true });
window.addEventListener("touchend", event => {
  const delta = touchY - event.changedTouches[0].clientY;
  if (Math.abs(delta) > 45) navigate(delta > 0 ? 1 : -1);
}, { passive: true });

window.addEventListener("keydown", event => {
  if (["ArrowDown", "PageDown", " "].includes(event.key)) navigate(1);
  if (["ArrowUp", "PageUp"].includes(event.key)) navigate(-1);
});

document.querySelector(".brand").addEventListener("click", event => {
  event.preventDefault();
  showScene(0);
});

function startAudio() {
  if (audio) return;
  audio = new AudioContext();
  const buffer = audio.createBuffer(1, audio.sampleRate * 2, audio.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  noiseSource = audio.createBufferSource();
  noiseSource.buffer = buffer;
  noiseSource.loop = true;
  const filter = audio.createBiquadFilter();
  filter.type = "lowpass";
  filter.frequency.value = 650;
  gain = audio.createGain();
  gain.gain.value = muted ? 0 : .04;
  noiseSource.connect(filter).connect(gain).connect(audio.destination);
  noiseSource.start();
}

function tuneAudio() {
  if (!audio || !gain) return;
  const levels = [.038, .07, .035, .055, .027];
  gain.gain.cancelScheduledValues(audio.currentTime);
  gain.gain.linearRampToValueAtTime(muted ? 0 : levels[active], audio.currentTime + .5);
}

soundButton.addEventListener("click", event => {
  event.stopPropagation();
  startAudio();
  muted = !muted;
  if (audio.state === "suspended") audio.resume();
  soundButton.classList.toggle("muted", muted);
  soundButton.setAttribute("aria-pressed", String(muted));
  soundButton.setAttribute("aria-label", muted ? "Turn on ambient sound" : "Mute ambient sound");
  soundButton.querySelector(".sound-label").textContent = muted ? "SOUND OFF" : "SOUND ON";
  tuneAudio();
});

document.querySelector("#experience").addEventListener("click", event => {
  if (event.target.closest("button, a")) return;
  startAudio();
  if (audio.state === "suspended") audio.resume();
  reaction.style.left = `${event.clientX}px`;
  reaction.style.top = `${event.clientY}px`;
  reaction.classList.remove("play");
  void reaction.offsetWidth;
  reaction.classList.add("play");
  const scene = document.querySelectorAll(".scene")[active];
  scene.classList.add("excited");
  setTimeout(() => scene.classList.remove("excited"), 950);
  clearTimeout(toastTimer);
  toast.textContent = `WILDLIFE RESPONSE · ${habitats[active].species[Math.floor(Math.random() * habitats[active].species.length)].toUpperCase()}`;
  toast.classList.add("show");
  toastTimer = setTimeout(() => toast.classList.remove("show"), 1400);
});

function resize() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  weather.width = innerWidth * ratio;
  weather.height = innerHeight * ratio;
  weather.style.width = `${innerWidth}px`;
  weather.style.height = `${innerHeight}px`;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  createParticles();
}

function createParticles() {
  const type = habitats[active].weather;
  const count = type === "rain" ? 170 : type === "current" ? 95 : 120;
  particles = Array.from({ length: count }, () => ({
    x: Math.random() * innerWidth,
    y: Math.random() * innerHeight,
    size: Math.random() * (type === "snow" ? 4 : 2) + .5,
    speed: Math.random() * 4 + 1,
    drift: Math.random() * 2 - 1,
    opacity: Math.random() * .5 + .15
  }));
}

function drawWeather() {
  const type = habitats[active].weather;
  context.clearRect(0, 0, innerWidth, innerHeight);
  particles.forEach(particle => {
    context.globalAlpha = particle.opacity;
    if (type === "rain") {
      context.strokeStyle = "#d7f1db";
      context.lineWidth = particle.size * .5;
      context.beginPath();
      context.moveTo(particle.x, particle.y);
      context.lineTo(particle.x - 7, particle.y + 24);
      context.stroke();
      particle.x -= 2.6;
      particle.y += particle.speed * 3.4;
    } else if (type === "snow") {
      context.fillStyle = "#fff";
      context.beginPath();
      context.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
      context.fill();
      particle.x += particle.drift;
      particle.y += particle.speed * .55;
    } else if (type === "current") {
      context.strokeStyle = "#a5f5ff";
      context.lineWidth = particle.size * .4;
      context.beginPath();
      context.moveTo(particle.x, particle.y);
      context.quadraticCurveTo(particle.x + 15, particle.y - 4, particle.x + 35, particle.y);
      context.stroke();
      particle.x += particle.speed * 1.7;
      particle.y += Math.sin(particle.x * .01) * .3;
    } else {
      context.fillStyle = type === "sand" ? "#ffd18a" : "#e6b36c";
      context.beginPath();
      context.arc(particle.x, particle.y, particle.size * .7, 0, Math.PI * 2);
      context.fill();
      particle.x += particle.speed * (type === "sand" ? 2.6 : .8);
      particle.y += particle.drift * .35;
    }
    if (particle.y > innerHeight + 30) particle.y = -30;
    if (particle.y < -30) particle.y = innerHeight + 30;
    if (particle.x > innerWidth + 40) particle.x = -40;
    if (particle.x < -40) particle.x = innerWidth + 40;
  });
  requestAnimationFrame(drawWeather);
}

window.addEventListener("resize", resize);
document.querySelector("#species").innerHTML = habitats[0].species.map(name => `<span>${name}</span>`).join("");
resize();
drawWeather();
const requestedScene = Number(new URLSearchParams(location.search).get("scene")) - 1;
if (requestedScene > 0 && requestedScene < habitats.length) showScene(requestedScene);
