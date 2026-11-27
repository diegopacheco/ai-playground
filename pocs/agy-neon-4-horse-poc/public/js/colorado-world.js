import * as THREE from 'three';
import {
  fbm2d,
  getMountainRidgeHeight,
  getTerrainHeight,
  getTrailCenterX,
  getTrailMask
} from './terrain-math.js';
import {
  createAspenBarkTexture,
  createAspenFoliageTexture,
  createCloudSpriteTexture,
  createDustSpriteTexture,
  createMeadowDetailTexture
} from './textures.js';

function createRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function createSkyDome() {
  const uniforms = {
    uZenith: { value: new THREE.Color('#1d66c2') },
    uHorizon: { value: new THREE.Color('#cfe8fc') },
    uGlow: { value: new THREE.Color('#fff0c8') },
    uSunDir: { value: new THREE.Vector3(0.5, 0.55, 0.65).normalize() }
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    vertexShader: `
      varying vec3 vWorldDir;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldDir = normalize(position);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: `
      uniform vec3 uZenith;
      uniform vec3 uHorizon;
      uniform vec3 uGlow;
      uniform vec3 uSunDir;
      varying vec3 vWorldDir;
      void main() {
        vec3 dir = normalize(vWorldDir);
        float h = clamp(dir.y, 0.0, 1.0);
        vec3 col = mix(uHorizon, uZenith, pow(h, 0.58));
        float sunDot = max(dot(dir, normalize(uSunDir)), 0.0);
        float halo = pow(sunDot, 12.0) * 0.38 + pow(sunDot, 96.0) * 0.62;
        float disc = smoothstep(0.9985, 0.9996, sunDot) * 1.4;
        float horizonGlow = pow(1.0 - h, 4.0) * pow(sunDot, 1.8) * 0.35;
        col += uGlow * (halo + disc + horizonGlow);
        gl_FragColor = vec4(col, 1.0);
      }
    `
  });

  const mesh = new THREE.Mesh(new THREE.SphereGeometry(520, 32, 20), material);
  return { mesh, uniforms };
}

function createMountainRing() {
  const radialSegs = 180;
  const ringSegs = 22;
  const innerRadius = 112;
  const outerRadius = 345;

  const positions = [];
  const colors = [];
  const indices = [];

  const colorForest = new THREE.Color('#345c39');
  const colorMaroon = new THREE.Color('#966052');
  const colorGranite = new THREE.Color('#9ea2ab');
  const colorSnow = new THREE.Color('#f7faff');
  const tmpColor = new THREE.Color();

  for (let r = 0; r <= ringSegs; r += 1) {
    const rn = r / ringSegs;
    const radius = innerRadius + rn * (outerRadius - innerRadius);
    for (let a = 0; a <= radialSegs; a += 1) {
      const an = a / radialSegs;
      const angle = an * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const ridgeH = getMountainRidgeHeight(angle, rn) * 0.92;
      const strata = Math.sin(ridgeH * 0.28 + angle * 6) * 2.0;
      const skirtDrop = rn < 0.08 ? -16 * (1 - rn / 0.08) : 0;
      const y = -5 + ridgeH + strata * rn * (1 - rn) + skirtDrop;
      positions.push(x, y, z);

      const noiseVal = fbm2d(x * 0.025, z * 0.025, 3);
      if (y < 20) {
        tmpColor.copy(colorForest).lerp(colorMaroon, Math.max(0, (y - 6) / 18) * 0.55);
      } else if (y < 46) {
        const t = (y - 20) / 26;
        tmpColor.copy(colorMaroon).lerp(colorGranite, t * 0.62 + noiseVal * 0.25);
      } else {
        const snowLine = 48 + (noiseVal - 0.5) * 16;
        const snowAmt = THREE.MathUtils.clamp((y - snowLine) / 11, 0, 1);
        tmpColor.copy(colorGranite).lerp(colorSnow, snowAmt);
      }
      colors.push(tmpColor.r, tmpColor.g, tmpColor.b);
    }
  }

  const stride = radialSegs + 1;
  for (let r = 0; r < ringSegs; r += 1) {
    for (let a = 0; a < radialSegs; a += 1) {
      const i0 = r * stride + a;
      const i1 = i0 + stride;
      const i2 = i1 + 1;
      const i3 = i0 + 1;
      indices.push(i0, i1, i3);
      indices.push(i3, i1, i2);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.9,
    metalness: 0.02,
    side: THREE.DoubleSide
  });
  return new THREE.Mesh(geo, mat);
}

function createGrassTuftGeometry() {
  const positions = [];
  const normals = [];
  const indices = [];
  const blades = [
    { ox: 0.0, oz: 0.0, rot: 0.0, tilt: 0.14, h: 0.52, w: 0.036 },
    { ox: 0.04, oz: 0.02, rot: 1.2, tilt: 0.22, h: 0.46, w: 0.032 },
    { ox: -0.04, oz: 0.03, rot: 2.5, tilt: 0.20, h: 0.42, w: 0.032 },
    { ox: 0.02, oz: -0.04, rot: 3.9, tilt: 0.18, h: 0.48, w: 0.030 },
    { ox: -0.03, oz: -0.03, rot: 5.1, tilt: 0.25, h: 0.36, w: 0.028 }
  ];

  let base = 0;
  for (const b of blades) {
    const cosR = Math.cos(b.rot);
    const sinR = Math.sin(b.rot);
    const dx = cosR * b.w;
    const dz = sinR * b.w;
    const bx = -sinR * b.tilt;
    const bz = cosR * b.tilt;

    positions.push(
      b.ox - dx, 0, b.oz - dz,
      b.ox + dx, 0, b.oz + dz,
      b.ox + bx * 0.4 - dx * 0.55, b.h * 0.55, b.oz + bz * 0.4 - dz * 0.55,
      b.ox + bx * 0.4 + dx * 0.55, b.h * 0.55, b.oz + bz * 0.4 + dz * 0.55,
      b.ox + bx, b.h, b.oz + bz
    );
    for (let k = 0; k < 5; k += 1) {
      normals.push(0, 1, 0);
    }
    indices.push(base, base + 1, base + 2);
    indices.push(base + 1, base + 3, base + 2);
    indices.push(base + 2, base + 3, base + 4);
    base += 5;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geo.setIndex(indices);
  return geo;
}

function createWildflowerGeometry() {
  const positions = [];
  const normals = [];
  const indices = [];
  let base = 0;

  const florets = [
    { ox: 0.0, oy: 0.34, oz: 0.0, r: 0.13 },
    { ox: 0.08, oy: 0.28, oz: 0.05, r: 0.10 },
    { ox: -0.07, oy: 0.26, oz: -0.04, r: 0.10 }
  ];

  for (const f of florets) {
    const petals = 5;
    for (let p = 0; p < petals; p += 1) {
      const a = (p / petals) * Math.PI * 2;
      const aL = a - 0.38;
      const aR = a + 0.38;
      positions.push(
        f.ox, f.oy - 0.015, f.oz,
        f.ox + Math.cos(aL) * f.r * 0.65, f.oy + 0.015, f.oz + Math.sin(aL) * f.r * 0.65,
        f.ox + Math.cos(a) * f.r, f.oy + 0.035, f.oz + Math.sin(a) * f.r,
        f.ox + Math.cos(aR) * f.r * 0.65, f.oy + 0.015, f.oz + Math.sin(aR) * f.r * 0.65
      );
      for (let k = 0; k < 4; k += 1) {
        normals.push(0, 1, 0);
      }
      indices.push(base, base + 1, base + 2);
      indices.push(base, base + 2, base + 3);
      base += 4;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geo.setIndex(indices);
  return geo;
}

function createScallopedPineCone(radius, height, radialSegs = 14) {
  const geo = new THREE.ConeGeometry(radius, height, radialSegs, 3, false);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i += 1) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const r = Math.hypot(x, z);
    if (r > 0.05) {
      const a = Math.atan2(z, x);
      const bough = 1 + Math.sin(a * 7) * 0.14 + Math.cos(a * 3) * 0.08;
      const droop = y < 0 ? -0.18 * (r / radius) : 0;
      pos.setXYZ(i, x * bough, y + droop, z * bough);
    }
  }
  geo.computeVertexNormals();
  return geo;
}

function createOrganicCanopyCluster() {
  const geo = new THREE.IcosahedronGeometry(1.35, 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i += 1) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const bump = 1 + 0.22 * Math.sin(x * 3.1 + y * 2.4) * Math.cos(z * 2.8);
    pos.setXYZ(i, x * bump, y * bump * 1.18, z * bump);
  }
  geo.computeVertexNormals();
  return geo;
}

function createClouds(cloudTex) {
  const group = new THREE.Group();
  const rng = createRng(1042);
  const clouds = [];

  for (let i = 0; i < 24; i += 1) {
    const mat = new THREE.SpriteMaterial({
      map: cloudTex,
      transparent: true,
      opacity: 0.82,
      depthWrite: false,
      fog: false
    });
    const sprite = new THREE.Sprite(mat);
    const angle = (i / 24) * Math.PI * 2 + (rng() - 0.5) * 0.25;
    const dist = 190 + rng() * 140;
    const y = 85 + rng() * 62;
    const scale = 38 + rng() * 44;
    sprite.position.set(Math.cos(angle) * dist, y, Math.sin(angle) * dist);
    sprite.scale.set(scale * 1.95, scale * 0.85, 1);
    group.add(sprite);
    clouds.push({ sprite, angle, dist, y, speed: 0.002 + rng() * 0.003 });
  }

  return { group, clouds };
}

function createHawks() {
  const group = new THREE.Group();
  const wingGeo = new THREE.BufferGeometry();
  wingGeo.setAttribute(
    'position',
    new THREE.Float32BufferAttribute([
      0, 0, 0.45,
      -1.4, 0.14, -0.1,
      0, 0, -0.35,
      0, 0, 0.45,
      0, 0, -0.35,
      1.4, 0.14, -0.1
    ], 3)
  );
  wingGeo.computeVertexNormals();
  const mat = new THREE.MeshBasicMaterial({ color: '#272320', side: THREE.DoubleSide });
  const birds = [];
  for (let i = 0; i < 2; i += 1) {
    const mesh = new THREE.Mesh(wingGeo, mat);
    mesh.scale.setScalar(1.35 + i * 0.2);
    group.add(mesh);
    birds.push({
      mesh,
      radius: 42 + i * 18,
      height: 36 + i * 12,
      phase: i * 2.8,
      speed: 0.18 + i * 0.04
    });
  }
  return { group, birds };
}

export function createColoradoWorld(scene) {
  const root = new THREE.Group();
  scene.add(root);

  const sky = createSkyDome();
  root.add(sky.mesh);

  const mountainRing = createMountainRing();
  root.add(mountainRing);

  const cloudTex = createCloudSpriteTexture();
  const cloudSystem = createClouds(cloudTex);
  root.add(cloudSystem.group);

  const hawkSystem = createHawks();
  root.add(hawkSystem.group);

  const terrainWidth = 240;
  const terrainDepth = 240;
  const segX = 112;
  const segZ = 112;
  const terrainGeo = new THREE.PlaneGeometry(terrainWidth, terrainDepth, segX, segZ);
  terrainGeo.rotateX(-Math.PI / 2);

  const vertCount = terrainGeo.attributes.position.count;
  const colorArray = new Float32Array(vertCount * 3);
  terrainGeo.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

  const detailTex = createMeadowDetailTexture();
  const terrainMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    map: detailTex,
    roughness: 0.92,
    metalness: 0.02
  });

  const terrainMesh = new THREE.Mesh(terrainGeo, terrainMat);
  terrainMesh.receiveShadow = true;
  root.add(terrainMesh);

  const baseX = new Float32Array(vertCount);
  const baseZ = new Float32Array(vertCount);
  const posAttr = terrainGeo.attributes.position;
  for (let i = 0; i < vertCount; i += 1) {
    baseX[i] = posAttr.getX(i);
    baseZ[i] = posAttr.getZ(i);
  }

  const colGrassLush = new THREE.Color('#518236');
  const colGrassSun = new THREE.Color('#82aa46');
  const colGrassDry = new THREE.Color('#ada252');
  const colDirt = new THREE.Color('#b07548');
  const colDirtRich = new THREE.Color('#8f5734');
  const colSlopeRock = new THREE.Color('#75736c');
  const scratchCol = new THREE.Color();

  const spanZ = 190;
  const rng = createRng(2026);

  function sampleSideOffset(minDist, maxDist) {
    const side = rng() > 0.5 ? 1 : -1;
    const dist = minDist + Math.pow(rng(), 0.72) * (maxDist - minDist);
    return side * dist;
  }

  const pineCount = 210;
  const pineData = [];
  for (let i = 0; i < pineCount; i += 1) {
    pineData.push({
      lateral: sampleSideOffset(11.5, 78),
      zOffset: (rng() - 0.5) * spanZ,
      scale: 0.78 + rng() * 1.18,
      rotY: rng() * Math.PI * 2
    });
  }

  const pineTrunkMesh = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.14, 0.28, 3.2, 7),
    new THREE.MeshStandardMaterial({ color: '#4d3728', roughness: 0.94 }),
    pineCount
  );
  pineTrunkMesh.castShadow = true;
  pineTrunkMesh.receiveShadow = true;

  const pineLayer1 = new THREE.InstancedMesh(
    createScallopedPineCone(2.25, 4.6, 14),
    new THREE.MeshStandardMaterial({ color: '#244b2f', roughness: 0.88 }),
    pineCount
  );
  pineLayer1.castShadow = true;
  pineLayer1.receiveShadow = true;

  const pineLayer2 = new THREE.InstancedMesh(
    createScallopedPineCone(1.75, 4.0, 14),
    new THREE.MeshStandardMaterial({ color: '#2b5836', roughness: 0.86 }),
    pineCount
  );
  pineLayer2.castShadow = true;

  const pineLayer3 = new THREE.InstancedMesh(
    createScallopedPineCone(1.22, 3.3, 12),
    new THREE.MeshStandardMaterial({ color: '#33673f', roughness: 0.84 }),
    pineCount
  );
  pineLayer3.castShadow = true;

  const pineLayer4 = new THREE.InstancedMesh(
    createScallopedPineCone(0.72, 2.4, 10),
    new THREE.MeshStandardMaterial({ color: '#3b7447', roughness: 0.82 }),
    pineCount
  );
  pineLayer4.castShadow = true;
  root.add(pineTrunkMesh, pineLayer1, pineLayer2, pineLayer3, pineLayer4);

  const aspenCount = 135;
  const aspenData = [];
  for (let i = 0; i < aspenCount; i += 1) {
    aspenData.push({
      lateral: sampleSideOffset(8.2, 54),
      zOffset: (rng() - 0.5) * spanZ,
      scale: 0.72 + rng() * 0.88,
      rotY: rng() * Math.PI * 2,
      phase: rng() * Math.PI * 2
    });
  }

  const aspenBarkTex = createAspenBarkTexture();
  const aspenFoliageTex = createAspenFoliageTexture();
  const aspenTrunkMesh = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.075, 0.145, 6.2, 8),
    new THREE.MeshStandardMaterial({ map: aspenBarkTex, roughness: 0.82 }),
    aspenCount
  );
  aspenTrunkMesh.castShadow = true;
  aspenTrunkMesh.receiveShadow = true;

  const canopyGeo = createOrganicCanopyCluster();
  const aspenCrownA = new THREE.InstancedMesh(
    canopyGeo,
    new THREE.MeshStandardMaterial({ map: aspenFoliageTex, color: '#9cc84e', roughness: 0.78 }),
    aspenCount
  );
  aspenCrownA.castShadow = true;

  const aspenCrownB = new THREE.InstancedMesh(
    canopyGeo,
    new THREE.MeshStandardMaterial({ map: aspenFoliageTex, color: '#b8db5c', roughness: 0.74 }),
    aspenCount
  );
  aspenCrownB.castShadow = true;
  root.add(aspenTrunkMesh, aspenCrownA, aspenCrownB);

  const grassCount = 2100;
  const grassData = [];
  const grassPalette = ['#5e8d34', '#729f3d', '#86ae45', '#9ab44c'];
  for (let i = 0; i < grassCount; i += 1) {
    grassData.push({
      lateral: sampleSideOffset(2.4, 34),
      zOffset: (rng() - 0.5) * 115,
      scale: 0.7 + rng() * 1.0,
      rotY: rng() * Math.PI * 2,
      phase: rng() * Math.PI * 2,
      color: new THREE.Color(grassPalette[Math.floor(rng() * grassPalette.length)])
    });
  }

  const grassMesh = new THREE.InstancedMesh(
    createGrassTuftGeometry(),
    new THREE.MeshStandardMaterial({ roughness: 0.84, side: THREE.DoubleSide }),
    grassCount
  );
  for (let i = 0; i < grassCount; i += 1) {
    grassMesh.setColorAt(i, grassData[i].color);
  }
  grassMesh.instanceColor.needsUpdate = true;
  grassMesh.receiveShadow = true;
  root.add(grassMesh);

  const flowerCount = 950;
  const flowerPalette = ['#7aa7fa', '#ea5234', '#8165e0', '#f5c736', '#f7f9ff'];
  const flowerData = [];
  for (let i = 0; i < flowerCount; i += 1) {
    flowerData.push({
      lateral: sampleSideOffset(2.5, 28),
      zOffset: (rng() - 0.5) * 110,
      scale: 0.75 + rng() * 0.9,
      rotY: rng() * Math.PI * 2,
      color: new THREE.Color(flowerPalette[Math.floor(rng() * flowerPalette.length)])
    });
  }

  const flowerMesh = new THREE.InstancedMesh(
    createWildflowerGeometry(),
    new THREE.MeshStandardMaterial({ roughness: 0.62, side: THREE.DoubleSide }),
    flowerCount
  );
  for (let i = 0; i < flowerCount; i += 1) {
    flowerMesh.setColorAt(i, flowerData[i].color);
  }
  flowerMesh.instanceColor.needsUpdate = true;
  root.add(flowerMesh);

  const rockCount = 72;
  const rockData = [];
  for (let i = 0; i < rockCount; i += 1) {
    rockData.push({
      lateral: sampleSideOffset(3.5, 42),
      zOffset: (rng() - 0.5) * spanZ,
      sx: 0.55 + rng() * 1.65,
      sy: 0.35 + rng() * 0.95,
      sz: 0.55 + rng() * 1.45,
      rotY: rng() * Math.PI * 2
    });
  }
  const rockMesh = new THREE.InstancedMesh(
    new THREE.DodecahedronGeometry(0.72, 0),
    new THREE.MeshStandardMaterial({ color: '#827c72', roughness: 0.95, flatShading: true }),
    rockCount
  );
  rockMesh.castShadow = true;
  rockMesh.receiveShadow = true;
  root.add(rockMesh);

  const fenceSpacing = 4.5;
  const fenceCount = 34;
  const fenceSpan = fenceCount * fenceSpacing;
  const fenceMat = new THREE.MeshStandardMaterial({ color: '#80664c', roughness: 0.92 });
  const fencePosts = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.075, 0.095, 1.38, 6), fenceMat, fenceCount);
  const fenceRailTop = new THREE.InstancedMesh(new THREE.BoxGeometry(0.085, 0.105, fenceSpacing + 0.18), fenceMat, fenceCount);
  const fenceRailLow = new THREE.InstancedMesh(new THREE.BoxGeometry(0.085, 0.105, fenceSpacing + 0.18), fenceMat, fenceCount);
  fencePosts.castShadow = true;
  fenceRailTop.castShadow = true;
  fenceRailLow.castShadow = true;
  root.add(fencePosts, fenceRailTop, fenceRailLow);

  const dustTex = createDustSpriteTexture();
  const dustPool = [];
  const dustGroup = new THREE.Group();
  root.add(dustGroup);
  for (let i = 0; i < 120; i += 1) {
    const mat = new THREE.SpriteMaterial({
      map: dustTex,
      transparent: true,
      opacity: 0,
      depthWrite: false
    });
    const sprite = new THREE.Sprite(mat);
    sprite.visible = false;
    dustGroup.add(sprite);
    dustPool.push({
      sprite,
      vel: new THREE.Vector3(),
      age: 99,
      maxAge: 1,
      startScale: 0.25,
      endScale: 1.5
    });
  }
  let dustCursor = 0;

  const pollenCount = 150;
  const pollenPos = new Float32Array(pollenCount * 3);
  for (let i = 0; i < pollenCount; i += 1) {
    pollenPos[i * 3] = (rng() - 0.5) * 32;
    pollenPos[i * 3 + 1] = 0.5 + rng() * 6.5;
    pollenPos[i * 3 + 2] = (rng() - 0.5) * 36;
  }
  const pollenGeo = new THREE.BufferGeometry();
  pollenGeo.setAttribute('position', new THREE.BufferAttribute(pollenPos, 3));
  const pollenPoints = new THREE.Points(
    pollenGeo,
    new THREE.PointsMaterial({
      color: '#fff6d6',
      size: 0.075,
      transparent: true,
      opacity: 0.52,
      depthWrite: false
    })
  );
  root.add(pollenPoints);

  const dummy = new THREE.Object3D();

  function wrapRelativeZ(rawWorldZ, worldZ, span) {
    const half = span * 0.5;
    let dz = (rawWorldZ - worldZ) % span;
    if (dz > half) dz -= span;
    if (dz < -half) dz += span;
    return dz;
  }

  function updateTerrainGeometry(worldZ, centerX) {
    const pos = terrainGeo.attributes.position;
    const col = terrainGeo.attributes.color;
    const snapStep = 2.0;
    const snappedZ = Math.floor(worldZ / snapStep) * snapStep;
    terrainMesh.position.set(0, 0, snappedZ - worldZ);
    detailTex.offset.y = -snappedZ * (26 / terrainDepth);

    for (let i = 0; i < vertCount; i += 1) {
      const lx = baseX[i];
      const lz = baseZ[i];
      const wx = lx + centerX;
      const wz = lz + snappedZ;
      const h = getTerrainHeight(wx, wz);
      pos.setXYZ(i, lx, h, lz);

      const trail = getTrailMask(wx, wz);
      const n = fbm2d(wx * 0.055, wz * 0.055, 3);
      scratchCol.copy(colGrassLush).lerp(colGrassSun, THREE.MathUtils.clamp(n * 1.25 - 0.15, 0, 1));
      if (n > 0.58) {
        scratchCol.lerp(colGrassDry, (n - 0.58) * 1.25);
      }
      if (h > 6.5) {
        scratchCol.lerp(colSlopeRock, THREE.MathUtils.clamp((h - 6.5) / 14, 0, 0.7));
      }
      if (trail > 0.01) {
        const dirtTone = colDirt.clone().lerp(colDirtRich, n * 0.5);
        scratchCol.lerp(dirtTone, trail);
      }
      col.setXYZ(i, scratchCol.r, scratchCol.g, scratchCol.b);
    }

    pos.needsUpdate = true;
    col.needsUpdate = true;
    terrainGeo.computeVertexNormals();
  }

  let lastTerrainSnap = Number.NaN;

  function update(worldZ, speedMps, dt, timeSec) {
    const centerX = getTrailCenterX(worldZ);
    const snapStep = 2.0;
    const snappedZ = Math.floor(worldZ / snapStep) * snapStep;

    if (snappedZ !== lastTerrainSnap) {
      updateTerrainGeometry(worldZ, centerX);
      lastTerrainSnap = snappedZ;
    } else {
      terrainMesh.position.z = snappedZ - worldZ;
    }

    mountainRing.position.set(0, -1.5, 0);

    for (const c of cloudSystem.clouds) {
      c.angle += c.speed * dt;
      c.sprite.position.set(Math.cos(c.angle) * c.dist, c.y, Math.sin(c.angle) * c.dist);
    }

    for (const b of hawkSystem.birds) {
      const a = timeSec * b.speed + b.phase;
      b.mesh.position.set(Math.cos(a) * b.radius, b.height + Math.sin(a * 2) * 2.5, Math.sin(a) * b.radius + 35);
      b.mesh.rotation.y = -a;
      b.mesh.rotation.z = 0.22;
    }

    for (let i = 0; i < pineCount; i += 1) {
      const item = pineData[i];
      const dz = wrapRelativeZ(item.zOffset, worldZ, spanZ);
      const wz = worldZ + dz;
      const wx = getTrailCenterX(wz) + item.lateral;
      const lx = wx - centerX;
      const y = getTerrainHeight(wx, wz);
      const s = item.scale;

      dummy.position.set(lx, y + 1.4 * s, dz);
      dummy.rotation.set(0, item.rotY, 0);
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      pineTrunkMesh.setMatrixAt(i, dummy.matrix);

      dummy.position.set(lx, y + 3.6 * s, dz);
      dummy.updateMatrix();
      pineLayer1.setMatrixAt(i, dummy.matrix);

      dummy.position.set(lx, y + 5.7 * s, dz);
      dummy.rotation.set(0, item.rotY + 0.5, 0);
      dummy.updateMatrix();
      pineLayer2.setMatrixAt(i, dummy.matrix);

      dummy.position.set(lx, y + 7.6 * s, dz);
      dummy.rotation.set(0, item.rotY + 1.1, 0);
      dummy.updateMatrix();
      pineLayer3.setMatrixAt(i, dummy.matrix);

      dummy.position.set(lx, y + 9.2 * s, dz);
      dummy.rotation.set(0, item.rotY + 1.7, 0);
      dummy.updateMatrix();
      pineLayer4.setMatrixAt(i, dummy.matrix);
    }
    pineTrunkMesh.instanceMatrix.needsUpdate = true;
    pineLayer1.instanceMatrix.needsUpdate = true;
    pineLayer2.instanceMatrix.needsUpdate = true;
    pineLayer3.instanceMatrix.needsUpdate = true;
    pineLayer4.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < aspenCount; i += 1) {
      const item = aspenData[i];
      const dz = wrapRelativeZ(item.zOffset, worldZ, spanZ);
      const wz = worldZ + dz;
      const wx = getTrailCenterX(wz) + item.lateral;
      const lx = wx - centerX;
      const y = getTerrainHeight(wx, wz);
      const s = item.scale;
      const sway = Math.sin(timeSec * 2.1 + item.phase) * 0.032;

      dummy.position.set(lx, y + 3.0 * s, dz);
      dummy.rotation.set(sway, item.rotY, sway * 0.7);
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      aspenTrunkMesh.setMatrixAt(i, dummy.matrix);

      dummy.position.set(lx + 0.16 * s, y + 5.6 * s, dz);
      dummy.scale.set(s * 1.12, s * 1.25, s * 1.02);
      dummy.updateMatrix();
      aspenCrownA.setMatrixAt(i, dummy.matrix);

      dummy.position.set(lx - 0.36 * s, y + 6.8 * s, dz + 0.22 * s);
      dummy.scale.set(s * 0.88, s * 1.0, s * 0.88);
      dummy.updateMatrix();
      aspenCrownB.setMatrixAt(i, dummy.matrix);
    }
    aspenTrunkMesh.instanceMatrix.needsUpdate = true;
    aspenCrownA.instanceMatrix.needsUpdate = true;
    aspenCrownB.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < grassCount; i += 1) {
      const item = grassData[i];
      const dz = wrapRelativeZ(item.zOffset, worldZ, 115);
      const wz = worldZ + dz;
      const wx = getTrailCenterX(wz) + item.lateral;
      const lx = wx - centerX;
      const y = getTerrainHeight(wx, wz);
      const wind = Math.sin(timeSec * 3.4 + item.phase + wz * 0.2) * 0.18;

      dummy.position.set(lx, y - 0.02, dz);
      dummy.rotation.set(wind, item.rotY, wind * 0.5);
      dummy.scale.set(item.scale, item.scale, item.scale);
      dummy.updateMatrix();
      grassMesh.setMatrixAt(i, dummy.matrix);
    }
    grassMesh.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < flowerCount; i += 1) {
      const item = flowerData[i];
      const dz = wrapRelativeZ(item.zOffset, worldZ, 110);
      const wz = worldZ + dz;
      const wx = getTrailCenterX(wz) + item.lateral;
      const lx = wx - centerX;
      const y = getTerrainHeight(wx, wz);

      dummy.position.set(lx, y, dz);
      dummy.rotation.set(0, item.rotY, 0);
      dummy.scale.set(item.scale, item.scale, item.scale);
      dummy.updateMatrix();
      flowerMesh.setMatrixAt(i, dummy.matrix);
    }
    flowerMesh.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < rockCount; i += 1) {
      const item = rockData[i];
      const dz = wrapRelativeZ(item.zOffset, worldZ, spanZ);
      const wz = worldZ + dz;
      const wx = getTrailCenterX(wz) + item.lateral;
      const lx = wx - centerX;
      const y = getTerrainHeight(wx, wz);

      dummy.position.set(lx, y + item.sy * 0.22, dz);
      dummy.rotation.set(0.2, item.rotY, 0.15);
      dummy.scale.set(item.sx, item.sy, item.sz);
      dummy.updateMatrix();
      rockMesh.setMatrixAt(i, dummy.matrix);
    }
    rockMesh.instanceMatrix.needsUpdate = true;

    for (let i = 0; i < fenceCount; i += 1) {
      const baseFenceZ = i * fenceSpacing;
      const dz = wrapRelativeZ(baseFenceZ, worldZ, fenceSpan);
      const wz = worldZ + dz;
      const wzNext = wz + fenceSpacing;
      const wx = getTrailCenterX(wz) - 7.4;
      const wxNext = getTrailCenterX(wzNext) - 7.4;
      const lx = wx - centerX;
      const y = getTerrainHeight(wx, wz);
      const yNext = getTerrainHeight(wxNext, wzNext);
      const yaw = Math.atan2(wxNext - wx, fenceSpacing);
      const pitch = -Math.atan2(yNext - y, fenceSpacing);

      dummy.position.set(lx, y + 0.64, dz);
      dummy.rotation.set(0, yaw, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      fencePosts.setMatrixAt(i, dummy.matrix);

      const midX = (wx + wxNext) * 0.5 - centerX;
      const midY = (y + yNext) * 0.5;
      const midZ = dz + fenceSpacing * 0.5;
      dummy.position.set(midX, midY + 1.05, midZ);
      dummy.rotation.set(pitch, yaw, 0);
      dummy.updateMatrix();
      fenceRailTop.setMatrixAt(i, dummy.matrix);

      dummy.position.set(midX, midY + 0.58, midZ);
      dummy.updateMatrix();
      fenceRailLow.setMatrixAt(i, dummy.matrix);
    }
    fencePosts.instanceMatrix.needsUpdate = true;
    fenceRailTop.instanceMatrix.needsUpdate = true;
    fenceRailLow.instanceMatrix.needsUpdate = true;

    for (const p of dustPool) {
      if (p.age >= p.maxAge) continue;
      p.age += dt;
      if (p.age >= p.maxAge) {
        p.sprite.visible = false;
        continue;
      }
      const t = p.age / p.maxAge;
      p.sprite.position.x += p.vel.x * dt;
      p.sprite.position.y += p.vel.y * dt;
      p.sprite.position.z += (p.vel.z - speedMps * 0.85) * dt;
      p.vel.y *= Math.max(0, 1 - dt * 1.4);
      const sc = THREE.MathUtils.lerp(p.startScale, p.endScale, Math.pow(t, 0.7));
      p.sprite.scale.set(sc, sc, 1);
      p.sprite.material.opacity = Math.sin(t * Math.PI) * 0.55;
    }

    const pAttr = pollenGeo.attributes.position;
    for (let i = 0; i < pollenCount; i += 1) {
      let pz = pAttr.getZ(i) - speedMps * dt * 0.55;
      let px = pAttr.getX(i) + Math.sin(timeSec * 0.9 + i) * dt * 0.35;
      let py = pAttr.getY(i) + Math.cos(timeSec * 1.3 + i) * dt * 0.18;
      if (pz < -18) pz += 36;
      if (px > 16) px -= 32;
      if (px < -16) px += 32;
      if (py < 0.4) py = 5.5;
      pAttr.setXYZ(i, px, py, pz);
    }
    pAttr.needsUpdate = true;
  }

  function spawnHoofDust(worldPos, intensity = 1) {
    const count = Math.max(1, Math.round(3 * intensity));
    for (let i = 0; i < count; i += 1) {
      const p = dustPool[dustCursor];
      dustCursor = (dustCursor + 1) % dustPool.length;
      p.age = 0;
      p.maxAge = 0.55 + Math.random() * 0.5;
      p.startScale = 0.24 + Math.random() * 0.16;
      p.endScale = (1.15 + Math.random() * 0.85) * (0.6 + intensity * 0.5);
      p.sprite.position.set(
        worldPos.x + (Math.random() - 0.5) * 0.18,
        worldPos.y + 0.12,
        worldPos.z + (Math.random() - 0.5) * 0.18
      );
      p.vel.set(
        (Math.random() - 0.5) * 1.1,
        0.75 + Math.random() * 1.05 * intensity,
        -1.2 - Math.random() * 1.4
      );
      p.sprite.scale.set(p.startScale, p.startScale, 1);
      p.sprite.material.opacity = 0.48;
      p.sprite.visible = true;
    }
  }

  function applyLightingMood(mood, sunDir) {
    sky.uniforms.uZenith.value.set(mood.skyZenith);
    sky.uniforms.uHorizon.value.set(mood.skyHorizon);
    sky.uniforms.uGlow.value.set(mood.skyGlow);
    sky.uniforms.uSunDir.value.copy(sunDir).normalize();
  }

  return {
    root,
    update,
    spawnHoofDust,
    applyLightingMood
  };
}
