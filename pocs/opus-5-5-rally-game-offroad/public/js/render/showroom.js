import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { buildCar } from './cars.js';

export function createShowroom(renderer) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#141312');
  scene.fog = new THREE.Fog('#141312', 14, 40);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();

  const floor = new THREE.Mesh(new THREE.CircleGeometry(30, 64), new THREE.MeshStandardMaterial({ color: '#1d1c1a', roughness: 0.35, metalness: 0.2 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);
  const plate = new THREE.Mesh(new THREE.CylinderGeometry(4.2, 4.3, 0.12, 64), new THREE.MeshStandardMaterial({ color: '#2a2825', roughness: 0.3, metalness: 0.6 }));
  plate.position.y = 0.06;
  plate.receiveShadow = true;
  scene.add(plate);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(4.25, 0.03, 8, 96), new THREE.MeshBasicMaterial({ color: '#ffb31a' }));
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.12;
  scene.add(ring);

  const key = new THREE.SpotLight('#fff1dc', 420, 40, 0.5, 0.6);
  key.position.set(6, 10, 6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  scene.add(key);
  const rim = new THREE.SpotLight('#9cc6ff', 260, 40, 0.6, 0.8);
  rim.position.set(-8, 6, -6);
  scene.add(rim);
  scene.add(new THREE.HemisphereLight('#ffffff', '#302a22', 0.4));

  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
  const turntable = new THREE.Group();
  turntable.position.y = 0.12;
  scene.add(turntable);
  let current = null;
  let angle = 0.7;

  return {
    scene,
    camera,
    show(spec, color, finish) {
      if (current) turntable.remove(current.root);
      current = buildCar(spec, color, finish);
      current.shadowBlob.visible = false;
      turntable.add(current.root);
    },
    update(dt, aspect) {
      angle += dt * 0.25;
      turntable.rotation.y = angle;
      camera.aspect = aspect;
      const wide = aspect > 1.1;
      camera.position.set(wide ? -1.6 : 0, 3.2, 16);
      camera.lookAt(wide ? -3.4 : 0, 1, 0);
      camera.updateProjectionMatrix();
    },
  };
}
