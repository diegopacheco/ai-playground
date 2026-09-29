import * as THREE from 'three';
import { TRACKS } from './core/tracks.js';
import { CARS } from './core/vehicle.js';
import { createWorld, createSim, stepSim, resetCarToTrack } from './core/sim.js';
import { wrapAngle, mulberry32 } from './core/math.js';
import { buildWorld } from './render/world.js';
import { buildCar, updateCarVisual, COLORS } from './render/cars.js';
import { createWeather, createSpray, createTireMarks } from './render/effects.js';

export const CAMERAS = ['Chase cam', 'Far chase', 'Hood cam', 'Bumper cam'];
const PLAYER = 3;

export function raceConfig(choice) {
  const rand = mulberry32(Date.now() & 0xffff);
  const pool = CARS.filter((c, k) => k !== choice.car).sort(() => rand() - 0.5);
  const colors = COLORS.filter((c) => c.hex !== choice.color).sort(() => rand() - 0.5);
  const cars = [pool[0], pool[1], pool[2], CARS[choice.car]];
  const paints = [
    { color: colors[0].hex, finish: 'Metallic' },
    { color: colors[1].hex, finish: 'Gloss' },
    { color: colors[2].hex, finish: 'Matte' },
    { color: choice.color, finish: choice.finish },
  ];
  return { ...choice, def: TRACKS[choice.track], cars, paints };
}

export function createRaceScene(renderer, config, quality) {
  const scene = new THREE.Scene();
  const world = createWorld(config.def);
  const sim = createSim({ world, weather: config.weather, specs: config.cars, playerIndex: PLAYER });
  const view = buildWorld(scene, renderer, world, config.weather, quality);
  const visuals = sim.cars.map((car, k) => {
    const v = buildCar(car.spec, config.paints[k].color, config.paints[k].finish);
    scene.add(v.root);
    return v;
  });
  const player = sim.cars[PLAYER];
  if (config.weather !== 'clear') {
    const lamp = new THREE.SpotLight('#fff2d6', 160, 90, 0.5, 0.5, 1.2);
    lamp.position.set(0, 1.3, player.spec.length / 2);
    lamp.target.position.set(0, 0, 40);
    visuals[PLAYER].root.add(lamp, lamp.target);
  }
  const weatherFx = createWeather(scene, config.weather, quality);
  const spray = createSpray(scene, quality);
  const marks = createTireMarks(scene, config.weather);
  const camera = new THREE.PerspectiveCamera(62, 1, 0.1, 9000);
  camera.userData.tmp = new THREE.Vector3();
  const heightAt = world.terrain.heightAt;
  let camMode = 0;
  let camHeading = player.heading;
  let shake = 0;
  let wrongWay = 0;
  const camPos = new THREE.Vector3(player.x, player.y + 4, player.z);
  const look = new THREE.Vector3();
  const target = new THREE.Vector3();

  function updateCamera(dt, aspect) {
    const car = player;
    const speed = Math.abs(car.u);
    const velHeading = speed > 3 ? Math.atan2(car.vx, car.vz) : car.heading;
    const blend = car.gear === -1 ? car.heading : car.heading + wrapAngle(velHeading - car.heading) * 0.35;
    camHeading += wrapAngle(blend - camHeading) * Math.min(1, dt * 4.5);
    const fx = Math.sin(camHeading);
    const fz = Math.cos(camHeading);
    const bump = (Math.random() - 0.5) * shake;
    shake = Math.max(0, shake - dt * 2.5);
    if (camMode <= 1) {
      const dist = camMode === 0 ? 7.2 + speed * 0.03 : 12 + speed * 0.04;
      const up = camMode === 0 ? 2.6 : 4.4;
      target.set(car.x - fx * dist, car.y + up, car.z - fz * dist);
      target.y = Math.max(target.y, heightAt(target.x, target.z) + 1.2);
      camPos.lerp(target, 1 - Math.exp(-dt * 7));
      camera.position.copy(camPos);
      camera.position.y += bump;
      look.set(car.x + fx * 4, car.y + 1.3, car.z + fz * 4);
      camera.lookAt(look);
      camera.fov = 60 + Math.min(18, speed * 0.32);
    } else {
      const root = visuals[PLAYER].root;
      const eyeY = camMode === 2 ? car.spec.height * 0.8 : 0.95;
      const eyeZ = camMode === 2 ? car.spec.length * 0.2 : car.spec.length / 2 + 0.25;
      root.updateMatrixWorld();
      camera.position.set(0, eyeY, eyeZ).applyMatrix4(root.matrixWorld);
      camera.position.y += bump * 0.4;
      const ahead = new THREE.Vector3(0, eyeY - 0.5, 30).applyMatrix4(root.matrixWorld);
      camera.lookAt(ahead);
      camera.fov = 72;
    }
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
  }

  return {
    scene,
    camera,
    sim,
    cycleCamera() {
      camMode = (camMode + 1) % CAMERAS.length;
      return CAMERAS[camMode];
    },
    cameraName: () => CAMERAS[camMode],
    reset() {
      resetCarToTrack(sim, PLAYER);
      camHeading = player.heading;
    },
    setShadows(size) {
      view.setShadows(size);
    },
    frame(dt, input, aspect, audio, hud) {
      const beforeLap = sim.race.entries[PLAYER].lap;
      stepSim(sim, dt, input);
      for (const ev of sim.events) {
        const near = ev.cars.includes(PLAYER);
        if (near) {
          shake = Math.min(0.5, shake + ev.strength * 0.05);
          audio?.impact(ev.strength);
        }
        spray.burst(ev.x, heightAt(ev.x, ev.z), ev.z, ev.strength, config.weather);
      }
      sim.cars.forEach((car, k) => {
        updateCarVisual(visuals[k], car, config.weather);
        spray.emitFromCar(car, config.weather, dt, heightAt);
      });
      spray.update(dt);
      marks.update(sim.cars, heightAt);
      updateCamera(dt, aspect);
      weatherFx.update(dt, camera);
      view.update(dt, visuals[PLAYER].root.position);
      const entry = sim.race.entries[PLAYER];
      if (entry.lap > beforeLap && !entry.finished) hud.message(entry.lap === sim.race.laps - 1 ? 'FINAL LAP' : `LAP ${entry.lap + 1}`, 'info', 2);
      const track = sim.world.track;
      const dir = wrapAngle(track.heading[player.proj.i] - player.heading);
      wrongWay = Math.abs(dir) > 2 && player.u > 3 ? wrongWay + dt : 0;
      if (wrongWay > 1) hud.message('WRONG WAY', 'warn', 0.3);
    },
  };
}
