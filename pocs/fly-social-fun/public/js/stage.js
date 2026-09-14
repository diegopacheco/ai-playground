import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { buildSpot, checkerTexture, swatterMesh } from './props.js';

const CLICK_TOLERANCE = 6;

function buildRoom(scene) {
  const cloth = checkerTexture();
  cloth.wrapS = cloth.wrapT = THREE.RepeatWrapping;
  cloth.repeat.set(3.25, 2);
  const table = new THREE.Mesh(new THREE.BoxGeometry(26, 0.6, 16), new THREE.MeshStandardMaterial({ map: cloth, roughness: 0.9 }));
  table.position.y = -0.3;
  table.receiveShadow = true;
  scene.add(table);

  const wood = new THREE.MeshStandardMaterial({ color: '#6b4428', roughness: 0.8 });
  for (const [x, z] of [[-12, -7], [12, -7], [-12, 7], [12, 7]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.8, 8, 0.8), wood);
    leg.position.set(x, -4.6, z);
    scene.add(leg);
  }

  const wallMaterial = new THREE.MeshStandardMaterial({ color: '#d7e6d8', roughness: 0.95 });
  const back = new THREE.Mesh(new THREE.PlaneGeometry(26, 22), wallMaterial);
  back.position.set(0, 5, -8);
  back.receiveShadow = true;
  scene.add(back);
  const left = new THREE.Mesh(new THREE.PlaneGeometry(16, 22), wallMaterial);
  left.position.set(-13, 5, 0);
  left.rotation.y = Math.PI / 2;
  left.receiveShadow = true;
  scene.add(left);

  const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: '#3a2c24', roughness: 1 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -8.6;
  scene.add(floor);
}

function buildLights(scene) {
  scene.add(new THREE.HemisphereLight('#fff4e0', '#4a3526', 1.1));
  const sun = new THREE.DirectionalLight('#fff1d6', 2.2);
  sun.position.set(8, 18, 12);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -16, right: 16, top: 16, bottom: -16, near: 1, far: 60 });
  scene.add(sun);
}

export function createStage(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  container.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#1d1a24');
  scene.fog = new THREE.Fog('#1d1a24', 45, 90);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
  camera.position.set(7, 12, 23);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 3.5, -1);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.minDistance = 3;
  controls.maxDistance = 45;

  buildRoom(scene);
  buildLights(scene);

  const spots = new Map();
  const animators = [];
  const effects = [];
  const frameListeners = [];
  const timer = new THREE.Timer();
  const raycaster = new THREE.Raycaster();
  let shake = 0;

  function buildSpots(list) {
    if (spots.size) return;
    for (const spot of list) {
      const built = buildSpot(spot);
      scene.add(built.group);
      spots.set(spot.id, { spot, group: built.group, position: new THREE.Vector3(...spot.position) });
      if (built.update) animators.push(built.update);
    }
  }

  function spotPosition(spotId) {
    return spots.get(spotId).position;
  }

  function addEffect(effect) {
    effects.push(effect);
  }

  function resize() {
    const { clientWidth: width, clientHeight: height } = container;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  function rayFrom(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    return raycaster;
  }

  function pickSpot(event, allowed) {
    const groups = [...spots.values()].filter(({ spot }) => allowed(spot)).map(({ group }) => group);
    const [hit] = rayFrom(event).intersectObjects(groups, true);
    return hit ? hit.object.userData.spotId : null;
  }

  function onClick(listener) {
    let down = null;
    renderer.domElement.addEventListener('pointerdown', (event) => (down = { x: event.clientX, y: event.clientY }));
    renderer.domElement.addEventListener('pointerup', (event) => {
      if (down && Math.hypot(event.clientX - down.x, event.clientY - down.y) < CLICK_TOLERANCE) listener(event);
      down = null;
    });
  }

  function focus(position) {
    const from = controls.target.clone();
    const to = position.clone();
    let t = 0;
    addEffect((dt) => {
      t = Math.min(1, t + dt / 0.8);
      controls.target.lerpVectors(from, to, 1 - Math.pow(1 - t, 3));
      return t < 1;
    });
  }

  function swat(spotId) {
    const target = spotPosition(spotId);
    const swatter = swatterMesh();
    scene.add(swatter);
    let t = 0;
    addEffect((dt) => {
      t += dt;
      const raise = target.y + 6;
      if (t < 0.35) {
        swatter.position.set(target.x, raise, target.z);
        swatter.rotation.x = -0.9 * (t / 0.35);
      } else if (t < 0.45) {
        const k = (t - 0.35) / 0.1;
        swatter.position.set(target.x, THREE.MathUtils.lerp(raise, target.y + 0.15, k), target.z);
        swatter.rotation.x = THREE.MathUtils.lerp(-0.9, 0, k);
        if (k > 0.8 && shake === 0) shake = 0.35;
      } else if (t > 1 && t < 1.4) {
        swatter.position.y = THREE.MathUtils.lerp(target.y + 0.15, raise + 4, (t - 1) / 0.4);
      }
      if (t < 1.4) return true;
      scene.remove(swatter);
      return false;
    });
  }

  function sparkle(position, color = '#fff6a8') {
    const count = 36;
    const velocities = Array.from({ length: count }, () => new THREE.Vector3(Math.random() - 0.5, Math.random(), Math.random() - 0.5).normalize().multiplyScalar(2 + Math.random() * 3));
    const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({ color, size: 0.18, transparent: true, depthWrite: false }));
    points.position.copy(position);
    scene.add(points);
    let t = 0;
    addEffect((dt) => {
      t += dt;
      const attribute = geometry.getAttribute('position');
      velocities.forEach((velocity, i) => attribute.setXYZ(i, velocity.x * t, velocity.y * t - 2 * t * t, velocity.z * t));
      attribute.needsUpdate = true;
      points.material.opacity = Math.max(0, 1 - t);
      if (t < 1) return true;
      scene.remove(points);
      geometry.dispose();
      points.material.dispose();
      return false;
    });
  }

  function snack(spotId, seconds) {
    const target = spotPosition(spotId);
    const colors = ['#c9853f', '#e0a458', '#8a5a2b', '#f2c36b'];
    const crumbs = Array.from({ length: 14 }, (_, i) => {
      const crumb = new THREE.Mesh(new THREE.DodecahedronGeometry(0.1 + Math.random() * 0.1), new THREE.MeshStandardMaterial({ color: colors[i % colors.length], flatShading: true }));
      crumb.userData.rest = new THREE.Vector3(target.x + (Math.random() - 0.5) * 1.6, target.y - 0.05, target.z + (Math.random() - 0.5) * 1.6);
      crumb.position.copy(crumb.userData.rest).add(new THREE.Vector3(0, 4 + Math.random() * 2, 0));
      crumb.castShadow = true;
      scene.add(crumb);
      return crumb;
    });
    sparkle(target.clone().add(new THREE.Vector3(0, 0.5, 0)), '#ffd66b');
    let t = 0;
    addEffect((dt) => {
      t += dt;
      for (const crumb of crumbs) {
        crumb.position.y = Math.max(crumb.userData.rest.y, crumb.position.y - dt * 9);
        crumb.rotation.x += dt * (crumb.position.y > crumb.userData.rest.y ? 8 : 0);
        if (t > seconds) crumb.scale.setScalar(Math.max(0.001, 1 - (t - seconds)));
      }
      if (t < seconds + 1) return true;
      crumbs.forEach((crumb) => {
        scene.remove(crumb);
        crumb.geometry.dispose();
        crumb.material.dispose();
      });
      return false;
    });
  }

  function onFrame(listener) {
    frameListeners.push(listener);
  }

  function loop() {
    timer.update();
    const dt = Math.min(timer.getDelta(), 0.05);
    const time = timer.getElapsed();
    animators.forEach((update) => update(dt, time));
    for (let i = effects.length - 1; i >= 0; i--) {
      if (!effects[i](dt, time)) effects.splice(i, 1);
    }
    frameListeners.forEach((listener) => listener(dt, time));
    controls.update();
    shake = Math.max(0, shake - dt);
    scene.position.set((Math.random() - 0.5) * shake, (Math.random() - 0.5) * shake, 0);
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  return { scene, camera, buildSpots, spotPosition, addEffect, rayFrom, pickSpot, onClick, focus, swat, sparkle, snack, onFrame };
}
