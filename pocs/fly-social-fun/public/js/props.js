import * as THREE from 'three';

function standard(color, extra = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.6, ...extra });
}

function mesh(geometry, material, position = [0, 0, 0]) {
  const result = new THREE.Mesh(geometry, material);
  result.position.set(...position);
  result.castShadow = true;
  result.receiveShadow = true;
  return result;
}

function canvasTexture(size, draw) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  draw(canvas.getContext('2d'), size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function banana() {
  const group = new THREE.Group();
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-1.7, 0.95, 0),
    new THREE.Vector3(-0.8, 0.3, 0),
    new THREE.Vector3(0.8, 0.3, 0),
    new THREE.Vector3(1.7, 0.95, 0),
  ]);
  const peel = standard('#f2cf3d', { roughness: 0.5 });
  group.add(mesh(new THREE.TubeGeometry(curve, 48, 0.42, 18), peel));
  const brown = standard('#5a3b1c');
  for (const end of [0, 1]) {
    const tip = mesh(new THREE.SphereGeometry(0.2, 10, 8), brown, curve.getPoint(end).toArray());
    group.add(tip);
  }
  for (const t of [0.28, 0.42, 0.55, 0.68, 0.8]) {
    const point = curve.getPoint(t).add(new THREE.Vector3(0, 0.38, (t - 0.5) * 0.3));
    const spot = mesh(new THREE.SphereGeometry(0.08, 8, 6), brown, point.toArray());
    spot.scale.set(1, 0.4, 1.3);
    group.add(spot);
  }
  group.position.y = -0.72;
  group.rotation.y = 0.5;
  return { group };
}

function pizza() {
  const group = new THREE.Group();
  const shape = new THREE.Shape();
  shape.moveTo(-1.5, 0);
  shape.lineTo(1.2, 1.05);
  shape.lineTo(1.2, -1.05);
  shape.closePath();
  const slice = new THREE.ExtrudeGeometry(shape, { depth: 0.16, bevelEnabled: true, bevelSize: 0.05, bevelThickness: 0.04, bevelSegments: 2 });
  slice.rotateX(-Math.PI / 2);
  group.add(mesh(slice, standard('#f6c14b', { roughness: 0.8 })));
  const crust = mesh(new THREE.CylinderGeometry(0.2, 0.2, 2.3, 14), standard('#c98a3c', { roughness: 0.9 }), [1.25, 0.12, 0]);
  crust.rotation.x = Math.PI / 2;
  group.add(crust);
  const pepperoni = standard('#b3372b', { roughness: 0.7 });
  for (const [x, z] of [[-0.4, 0], [0.4, 0.45], [0.5, -0.4], [0.9, 0.05]]) {
    group.add(mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.05, 16), pepperoni, [x, 0.23, z]));
  }
  group.position.y = -0.2;
  group.rotation.y = -0.6;
  return { group };
}

function trash() {
  const group = new THREE.Group();
  const metal = standard('#8d959c', { metalness: 0.65, roughness: 0.35 });
  group.add(mesh(new THREE.CylinderGeometry(1.3, 1.1, 3.2, 32), metal, [0, 1.6, 0]));
  for (const y of [0.7, 1.6, 2.5]) {
    const rib = mesh(new THREE.TorusGeometry(1.22 + (y - 1.6) * 0.06, 0.04, 8, 40), metal, [0, y, 0]);
    rib.rotation.x = Math.PI / 2;
    group.add(rib);
  }
  const lid = mesh(new THREE.CylinderGeometry(1.38, 1.38, 0.12, 32), metal, [0.35, 3.45, 0]);
  lid.rotation.z = 0.35;
  group.add(lid);
  group.add(mesh(new THREE.IcosahedronGeometry(0.4, 0), standard('#f1efe6', { flatShading: true }), [-0.5, 3.2, 0.3]));
  const peel = mesh(new THREE.TorusGeometry(0.35, 0.1, 8, 16, Math.PI), standard('#e7c43a'), [-0.9, 3.1, -0.4]);
  peel.rotation.set(0.4, 0.3, 1.2);
  group.add(peel);

  const stinkMaterial = new THREE.MeshBasicMaterial({ color: '#9bd45a', transparent: true, opacity: 0.35, depthWrite: false });
  const puffs = Array.from({ length: 12 }, (_, i) => {
    const puff = new THREE.Mesh(new THREE.SphereGeometry(0.16, 8, 6), stinkMaterial);
    puff.userData.offset = i / 12;
    group.add(puff);
    return puff;
  });
  function update(dt, time) {
    for (const puff of puffs) {
      const t = (time * 0.25 + puff.userData.offset) % 1;
      const lane = puff.userData.offset * Math.PI * 2;
      puff.position.set(Math.cos(lane) * 0.6 + Math.sin(time * 2 + lane) * 0.25, 3.5 + t * 3, Math.sin(lane) * 0.6);
      puff.scale.setScalar(1.2 - t);
    }
  }
  group.position.y = -3.5;
  return { group, update };
}

function mug() {
  const group = new THREE.Group();
  const ceramic = standard('#3f7cac', { roughness: 0.35 });
  group.add(mesh(new THREE.CylinderGeometry(0.9, 0.82, 1.9, 32, 1, true), new THREE.MeshStandardMaterial({ color: '#3f7cac', roughness: 0.35, side: THREE.DoubleSide }), [0, 0.95, 0]));
  group.add(mesh(new THREE.CylinderGeometry(0.82, 0.82, 0.08, 32), ceramic, [0, 0.04, 0]));
  group.add(mesh(new THREE.CylinderGeometry(0.86, 0.86, 0.04, 32), standard('#3a2214', { roughness: 0.2 }), [0, 1.35, 0]));
  const handle = mesh(new THREE.TorusGeometry(0.45, 0.12, 12, 24, Math.PI * 1.2), ceramic, [0.9, 0.95, 0]);
  handle.rotation.z = -Math.PI * 0.6;
  group.add(handle);
  const rim = mesh(new THREE.TorusGeometry(0.9, 0.04, 8, 40), standard('#f4f1e8'), [0, 1.9, 0]);
  rim.rotation.x = Math.PI / 2;
  group.add(rim);
  group.position.y = -2.1;
  return { group };
}

function lamp() {
  const group = new THREE.Group();
  const brass = standard('#c9a14a', { metalness: 0.7, roughness: 0.3 });
  group.add(mesh(new THREE.CylinderGeometry(1.1, 1.2, 0.25, 32), brass, [0, 0.12, 0]));
  group.add(mesh(new THREE.CylinderGeometry(0.08, 0.08, 6.2, 12), brass, [0, 3.2, 0]));
  const shade = mesh(new THREE.ConeGeometry(1.5, 1.6, 32, 1, true), new THREE.MeshStandardMaterial({ color: '#e8574a', roughness: 0.7, side: THREE.DoubleSide }), [0, 6.4, 0]);
  group.add(shade);
  const bulb = mesh(new THREE.SphereGeometry(0.38, 20, 16), new THREE.MeshStandardMaterial({ color: '#fff3c4', emissive: '#ffd66b', emissiveIntensity: 3 }), [0, 5.75, 0]);
  bulb.castShadow = false;
  group.add(bulb);
  const light = new THREE.PointLight('#ffcf73', 40, 20, 1.6);
  light.position.set(0, 5.6, 0);
  group.add(light);
  function update(dt, time) {
    light.intensity = 38 + Math.sin(time * 9) * 1.5;
  }
  group.position.y = -5.6;
  return { group, update };
}

function windowPane() {
  const group = new THREE.Group();
  const sky = canvasTexture(512, (ctx, size) => {
    const gradient = ctx.createLinearGradient(0, 0, 0, size);
    gradient.addColorStop(0, '#6fb7ff');
    gradient.addColorStop(1, '#d9f0ff');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    for (const [x, y, r] of [[120, 120, 40], [170, 110, 55], [220, 125, 38], [360, 200, 34], [400, 190, 46], [440, 205, 30]]) {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = '#4f9a4a';
    ctx.fillRect(0, size * 0.8, size, size * 0.2);
    ctx.fillStyle = '#3d7a3a';
    ctx.beginPath();
    ctx.arc(90, size * 0.78, 70, 0, Math.PI * 2);
    ctx.arc(420, size * 0.8, 90, 0, Math.PI * 2);
    ctx.fill();
  });
  group.add(new THREE.Mesh(new THREE.PlaneGeometry(7, 5), new THREE.MeshBasicMaterial({ map: sky })));
  const wood = standard('#f3efe6', { roughness: 0.8 });
  for (const [w, h, x, y] of [[7.6, 0.3, 0, 2.6], [7.6, 0.3, 0, -2.6], [0.3, 5.5, -3.65, 0], [0.3, 5.5, 3.65, 0], [0.16, 5, 0, 0], [7, 0.16, 0, 0]]) {
    group.add(mesh(new THREE.BoxGeometry(w, h, 0.25), wood, [x, y, 0.1]));
  }
  group.add(mesh(new THREE.BoxGeometry(8.2, 0.25, 0.9), wood, [0, -2.85, 0.35]));
  group.position.set(0, 0, -0.52);
  return { group };
}

function web() {
  const group = new THREE.Group();
  const thread = new THREE.LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.55 });
  const radials = [];
  for (let i = 0; i < 14; i++) {
    const angle = (i / 14) * Math.PI * 2;
    radials.push(0, 0, 0, Math.cos(angle) * 2.8, Math.sin(angle) * 2.8, 0);
  }
  group.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(radials, 3)), thread));
  const spiral = [];
  for (let a = 0; a < Math.PI * 14; a += 0.12) {
    const r = 0.25 + (a / (Math.PI * 14)) * 2.5;
    spiral.push(Math.cos(a) * r, Math.sin(a) * r, 0);
  }
  group.add(new THREE.Line(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(spiral, 3)), thread));

  const spider = new THREE.Group();
  const black = standard('#161318', { roughness: 0.4 });
  spider.add(mesh(new THREE.SphereGeometry(0.42, 16, 12), black, [0, -0.3, 0]));
  spider.add(mesh(new THREE.SphereGeometry(0.26, 16, 12), black, [0, 0.25, 0]));
  const eyeMaterial = new THREE.MeshBasicMaterial({ color: '#ff3030' });
  for (const x of [-0.09, 0.09]) spider.add(mesh(new THREE.SphereGeometry(0.05, 8, 6), eyeMaterial, [x, 0.35, 0.22]));
  const legGeometry = new THREE.CylinderGeometry(0.03, 0.02, 0.9, 5);
  for (let i = 0; i < 8; i++) {
    const side = i < 4 ? -1 : 1;
    const leg = mesh(legGeometry, black, [side * 0.4, 0.25 - (i % 4) * 0.18, 0]);
    leg.rotation.z = side * (1.1 + (i % 4) * 0.25);
    spider.add(leg);
  }
  spider.position.set(0.9, 1.1, 0.1);
  group.add(spider);

  function update(dt, time) {
    spider.position.y = 1.1 + Math.sin(time * 0.8) * 0.25;
    spider.rotation.z = Math.sin(time * 0.6) * 0.15;
  }
  group.rotation.y = Math.PI / 4;
  return { group, update };
}

const BUILDERS = { banana, pizza, trash, mug, lamp, window: windowPane, web };

export function buildSpot(spot) {
  const builder = BUILDERS[spot.id];
  const built = builder ? builder() : { group: new THREE.Group() };
  const holder = new THREE.Group();
  holder.position.set(...spot.position);
  holder.add(built.group);
  holder.traverse((child) => (child.userData.spotId = spot.id));
  return { group: holder, update: built.update };
}

export function checkerTexture() {
  return canvasTexture(512, (ctx, size) => {
    const cells = 8;
    const cell = size / cells;
    for (let x = 0; x < cells; x++) {
      for (let y = 0; y < cells; y++) {
        ctx.fillStyle = (x + y) % 2 ? '#f3ead6' : '#c8463a';
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
    }
    ctx.strokeStyle = 'rgba(0,0,0,0.08)';
    for (let i = 0; i <= cells; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cell, 0);
      ctx.lineTo(i * cell, size);
      ctx.moveTo(0, i * cell);
      ctx.lineTo(size, i * cell);
      ctx.stroke();
    }
  });
}

export function swatterMesh() {
  const group = new THREE.Group();
  const texture = canvasTexture(256, (ctx, size) => {
    ctx.fillStyle = '#e8443a';
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#9e241c';
    for (let x = 8; x < size; x += 24) {
      for (let y = 8; y < size; y += 24) ctx.fillRect(x, y, 12, 12);
    }
  });
  group.add(mesh(new THREE.BoxGeometry(3, 0.1, 3.4), new THREE.MeshStandardMaterial({ map: texture, roughness: 0.8 })));
  group.add(mesh(new THREE.CylinderGeometry(0.1, 0.13, 7, 10), standard('#f2d14b'), [0, 0, 5.2]));
  group.children[1].rotation.x = Math.PI / 2;
  return group;
}
