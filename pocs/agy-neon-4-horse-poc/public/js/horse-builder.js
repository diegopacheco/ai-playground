import * as THREE from 'three';
import { COATS } from './presets.js';
import { createCoatTextures, createHairTexture } from './textures.js';

function createLoftedAlongZ(sections, radialSegs = 26) {
  const positions = [];
  const uvs = [];
  const indices = [];
  const ringCount = sections.length;
  const increasingZ = sections[ringCount - 1].z >= sections[0].z;

  for (let i = 0; i < ringCount; i += 1) {
    const s = sections[i];
    const v = i / (ringCount - 1);
    const rx = s.rx;
    const ry = s.ry;
    const cx = s.x || 0;
    const cy = s.y || 0;
    const cz = s.z || 0;
    const topPinch = s.topPinch || 0;
    const bottomPinch = s.bottomPinch || 0;
    const cleft = s.cleft || 0;

    for (let j = 0; j <= radialSegs; j += 1) {
      const u = j / radialSegs;
      const theta = u * Math.PI * 2;
      const sinT = Math.sin(theta);
      const cosT = Math.cos(theta);
      let px = sinT * rx;
      let py = cosT * ry;

      if (cosT > 0 && topPinch > 0) {
        px *= 1 - topPinch * Math.pow(cosT, 1.5);
      }
      if (cosT < 0 && bottomPinch > 0) {
        px *= 1 - bottomPinch * Math.pow(-cosT, 1.4);
      }
      if (cleft > 0 && Math.abs(sinT) < 0.45 && cosT > -0.2) {
        const notch = Math.cos((Math.abs(sinT) / 0.45) * Math.PI * 0.5);
        py -= cleft * notch * Math.max(0, cosT + 0.2);
      }

      positions.push(cx + px, cy + py, cz);
      uvs.push(u, v);
    }
  }

  const stride = radialSegs + 1;
  for (let i = 0; i < ringCount - 1; i += 1) {
    for (let j = 0; j < radialSegs; j += 1) {
      const a = i * stride + j;
      const b = a + stride;
      const c = b + 1;
      const d = a + 1;
      if (increasingZ) {
        indices.push(a, b, d);
        indices.push(d, b, c);
      } else {
        indices.push(a, d, b);
        indices.push(d, c, b);
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

function createLoftedLimb(sections, radialSegs = 18) {
  const positions = [];
  const uvs = [];
  const indices = [];
  const ringCount = sections.length;
  const increasingY = sections[ringCount - 1].y >= sections[0].y;

  for (let i = 0; i < ringCount; i += 1) {
    const s = sections[i];
    const v = i / (ringCount - 1);
    const rx = s.rx;
    const rz = s.rz;
    const cx = s.x || 0;
    const cy = s.y || 0;
    const cz = s.z || 0;
    const backRidge = s.backRidge || 0;

    for (let j = 0; j <= radialSegs; j += 1) {
      const u = j / radialSegs;
      const theta = u * Math.PI * 2;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);
      let px = cosT * rx;
      let pz = sinT * rz;
      if (sinT < -0.35 && backRidge > 0) {
        pz -= backRidge * Math.pow(-sinT - 0.35, 1.2);
      }
      positions.push(cx + px, cy, cz + pz);
      uvs.push(u, v);
    }
  }

  const stride = radialSegs + 1;
  for (let i = 0; i < ringCount - 1; i += 1) {
    for (let j = 0; j < radialSegs; j += 1) {
      const a = i * stride + j;
      const b = a + stride;
      const c = b + 1;
      const d = a + 1;
      if (increasingY) {
        indices.push(a, b, d);
        indices.push(d, b, c);
      } else {
        indices.push(a, d, b);
        indices.push(d, c, b);
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

function createHairRibbonGeometry(width, length, curveDrop = 0.08, segsY = 8) {
  const geo = new THREE.PlaneGeometry(width, length, 2, segsY);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i += 1) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const t = Math.max(0, Math.min(1, 0.5 - y / length));
    const taper = 1 - Math.pow(t, 1.35) * 0.82;
    const archZ = -Math.sin(t * Math.PI * 0.85) * curveDrop;
    const curlX = Math.sin(t * Math.PI) * width * 0.12;
    pos.setXYZ(i, x * taper + curlX, -t * length, archZ);
  }
  geo.computeVertexNormals();
  return geo;
}

function buildThoraxMesh(material) {
  const sections = [
    { z: 0.90, y: 0.01, rx: 0.015, ry: 0.015 },
    { z: 0.84, y: 0.02, rx: 0.165, ry: 0.195, bottomPinch: 0.24 },
    { z: 0.74, y: 0.04, rx: 0.245, ry: 0.285, bottomPinch: 0.20 },
    { z: 0.60, y: 0.08, rx: 0.285, ry: 0.335, topPinch: 0.24, bottomPinch: 0.14 },
    { z: 0.44, y: 0.11, rx: 0.305, ry: 0.355, topPinch: 0.32, bottomPinch: 0.10 },
    { z: 0.26, y: 0.08, rx: 0.315, ry: 0.34, topPinch: 0.22, bottomPinch: 0.08 },
    { z: 0.08, y: 0.05, rx: 0.302, ry: 0.315, topPinch: 0.16, bottomPinch: 0.11 },
    { z: -0.10, y: 0.045, rx: 0.282, ry: 0.292, topPinch: 0.14, bottomPinch: 0.15 }
  ];
  const mesh = new THREE.Mesh(createLoftedAlongZ(sections, 30), material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildHindquartersMesh(material) {
  const sections = [
    { z: 0.12, y: 0.05, rx: 0.294, ry: 0.306, topPinch: 0.15, bottomPinch: 0.12 },
    { z: -0.04, y: 0.045, rx: 0.278, ry: 0.288, topPinch: 0.13, bottomPinch: 0.18 },
    { z: -0.22, y: 0.068, rx: 0.284, ry: 0.29, topPinch: 0.10, bottomPinch: 0.25, cleft: 0.018 },
    { z: -0.44, y: 0.105, rx: 0.312, ry: 0.32, topPinch: 0.08, bottomPinch: 0.22, cleft: 0.036 },
    { z: -0.65, y: 0.085, rx: 0.292, ry: 0.302, topPinch: 0.11, bottomPinch: 0.20, cleft: 0.044 },
    { z: -0.82, y: 0.025, rx: 0.215, ry: 0.225, topPinch: 0.16, bottomPinch: 0.18, cleft: 0.032 },
    { z: -0.92, y: -0.02, rx: 0.10, ry: 0.115 },
    { z: -0.96, y: -0.04, rx: 0.015, ry: 0.015 }
  ];
  const mesh = new THREE.Mesh(createLoftedAlongZ(sections, 30), material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildLowerNeckMesh(material) {
  const sections = [
    { z: -0.18, y: -0.09, rx: 0.245, ry: 0.305, topPinch: 0.32, bottomPinch: 0.18 },
    { z: 0.00, y: 0.02, rx: 0.205, ry: 0.275, topPinch: 0.38, bottomPinch: 0.22 },
    { z: 0.18, y: 0.16, rx: 0.165, ry: 0.235, topPinch: 0.42, bottomPinch: 0.24 },
    { z: 0.35, y: 0.29, rx: 0.135, ry: 0.195, topPinch: 0.44, bottomPinch: 0.26 }
  ];
  const mesh = new THREE.Mesh(createLoftedAlongZ(sections, 24), material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildUpperNeckMesh(material) {
  const sections = [
    { z: -0.08, y: -0.06, rx: 0.142, ry: 0.202, topPinch: 0.42, bottomPinch: 0.26 },
    { z: 0.08, y: 0.08, rx: 0.118, ry: 0.168, topPinch: 0.44, bottomPinch: 0.28 },
    { z: 0.22, y: 0.19, rx: 0.102, ry: 0.140, topPinch: 0.38, bottomPinch: 0.26 },
    { z: 0.33, y: 0.27, rx: 0.092, ry: 0.122, topPinch: 0.28, bottomPinch: 0.22 }
  ];
  const mesh = new THREE.Mesh(createLoftedAlongZ(sections, 22), material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildHeadGroup(materials) {
  const head = new THREE.Group();

  const skullSections = [
    { z: -0.10, y: 0.03, rx: 0.015, ry: 0.015 },
    { z: -0.05, y: 0.04, rx: 0.106, ry: 0.132, topPinch: 0.22 },
    { z: 0.06, y: 0.028, rx: 0.126, ry: 0.156, topPinch: 0.20, bottomPinch: 0.22 },
    { z: 0.16, y: -0.028, rx: 0.108, ry: 0.134, topPinch: 0.22, bottomPinch: 0.24 },
    { z: 0.27, y: -0.102, rx: 0.082, ry: 0.102, topPinch: 0.24, bottomPinch: 0.22 },
    { z: 0.37, y: -0.165, rx: 0.072, ry: 0.086, topPinch: 0.18, bottomPinch: 0.18 },
    { z: 0.45, y: -0.202, rx: 0.068, ry: 0.078, topPinch: 0.10, bottomPinch: 0.12 },
    { z: 0.51, y: -0.218, rx: 0.042, ry: 0.048 },
    { z: 0.535, y: -0.224, rx: 0.008, ry: 0.008 }
  ];
  const skull = new THREE.Mesh(createLoftedAlongZ(skullSections, 24), materials.coat);
  skull.castShadow = true;
  skull.receiveShadow = true;
  head.add(skull);

  const muzzleGeo = new THREE.SphereGeometry(0.074, 18, 16);
  muzzleGeo.scale(0.96, 0.94, 1.18);
  const muzzle = new THREE.Mesh(muzzleGeo, materials.muzzle);
  muzzle.position.set(0, -0.208, 0.458);
  muzzle.castShadow = true;
  head.add(muzzle);

  const jawGeo = new THREE.SphereGeometry(0.098, 16, 14);
  jawGeo.scale(0.62, 0.94, 1.12);
  for (const side of [-1, 1]) {
    const cheek = new THREE.Mesh(jawGeo, materials.coat);
    cheek.position.set(side * 0.052, -0.055, 0.065);
    cheek.rotation.z = side * 0.14;
    cheek.castShadow = true;
    head.add(cheek);

    const brow = new THREE.Mesh(new THREE.SphereGeometry(0.030, 12, 10), materials.coat);
    brow.scale.set(0.82, 0.72, 1.35);
    brow.position.set(side * 0.098, 0.028, 0.148);
    head.add(brow);

    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.023, 14, 12), materials.eye);
    eye.position.set(side * 0.102, 0.005, 0.156);
    head.add(eye);

    const nostril = new THREE.Mesh(new THREE.SphereGeometry(0.017, 10, 10), materials.darkCavity);
    nostril.scale.set(0.62, 0.94, 1.48);
    nostril.position.set(side * 0.050, -0.19, 0.468);
    nostril.rotation.y = side * 0.32;
    head.add(nostril);
  }

  const blazeGeo = new THREE.CylinderGeometry(0.014, 0.024, 0.28, 8);
  blazeGeo.scale(1.15, 1, 0.35);
  const blaze = new THREE.Mesh(blazeGeo, materials.whiteMarking);
  blaze.position.set(0, -0.025, 0.265);
  blaze.rotation.x = Math.PI * 0.5 + 0.54;
  blaze.name = 'blaze';
  head.add(blaze);

  const earGeo = createLoftedLimb([
    { y: 0.0, rx: 0.030, rz: 0.036 },
    { y: 0.055, rx: 0.034, rz: 0.030 },
    { y: 0.125, rx: 0.022, rz: 0.018, z: 0.012 },
    { y: 0.17, rx: 0.006, rz: 0.006, z: 0.022 }
  ], 14);

  const earLeft = new THREE.Group();
  earLeft.position.set(0.072, 0.128, 0.018);
  earLeft.rotation.set(0.32, 0.18, -0.24);
  const earLeftMesh = new THREE.Mesh(earGeo, materials.coat);
  earLeftMesh.castShadow = true;
  earLeft.add(earLeftMesh);
  head.add(earLeft);

  const earRight = new THREE.Group();
  earRight.position.set(-0.072, 0.128, 0.018);
  earRight.rotation.set(0.32, -0.18, 0.24);
  const earRightMesh = new THREE.Mesh(earGeo, materials.coat);
  earRightMesh.castShadow = true;
  earRight.add(earRightMesh);
  head.add(earRight);

  const forelock = new THREE.Mesh(createHairRibbonGeometry(0.09, 0.24, 0.05, 6), materials.hair);
  forelock.position.set(0, 0.145, 0.07);
  forelock.rotation.set(-0.85, 0, 0);
  head.add(forelock);

  return { head, earLeft, earRight, blaze, forelock };
}

function buildHoofGroup(materials) {
  const hoofPivot = new THREE.Group();
  const coronet = new THREE.Mesh(
    createLoftedLimb([
      { y: 0.0, rx: 0.044, rz: 0.052 },
      { y: -0.038, rx: 0.053, rz: 0.063, z: 0.012 },
      { y: -0.105, rx: 0.061, rz: 0.072, z: 0.024 },
      { y: -0.132, rx: 0.062, rz: 0.073, z: 0.026 }
    ], 16),
    materials.hoof
  );
  coronet.castShadow = true;
  coronet.receiveShadow = true;
  hoofPivot.add(coronet);

  const sole = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.058, 0.012, 16), materials.hoof);
  sole.position.set(0, -0.132, 0.026);
  hoofPivot.add(sole);

  return hoofPivot;
}

function buildForeLeg(side, hasSock, materials) {
  const upperPivot = new THREE.Group();
  upperPivot.position.set(side * 0.19, 0.0, 0.56);

  const upperMesh = new THREE.Mesh(
    createLoftedLimb([
      { y: 0.19, rx: 0.02, rz: 0.03, x: -side * 0.06, z: -0.02 },
      { y: 0.12, rx: 0.105, rz: 0.158, x: -side * 0.03, z: -0.02 },
      { y: 0.00, rx: 0.106, rz: 0.145, z: -0.02 },
      { y: -0.14, rx: 0.094, rz: 0.126, z: -0.028 },
      { y: -0.29, rx: 0.076, rz: 0.098, z: -0.018 },
      { y: -0.41, rx: 0.059, rz: 0.074, z: -0.006 },
      { y: -0.50, rx: 0.052, rz: 0.062, z: 0.0 }
    ], 18),
    materials.coat
  );
  upperMesh.castShadow = true;
  upperMesh.receiveShadow = true;
  upperPivot.add(upperMesh);

  const lowerPivot = new THREE.Group();
  lowerPivot.position.set(0, -0.50, 0);
  upperPivot.add(lowerPivot);

  const kneeJoint = new THREE.Mesh(new THREE.SphereGeometry(0.054, 14, 12), materials.points);
  kneeJoint.scale.set(0.95, 1.05, 1.06);
  kneeJoint.castShadow = true;
  lowerPivot.add(kneeJoint);

  const cannonMat = hasSock ? materials.whiteMarking : materials.points;
  const upperCannon = new THREE.Mesh(
    createLoftedLimb([
      { y: 0.0, rx: 0.048, rz: 0.058 },
      { y: -0.14, rx: 0.038, rz: 0.047, backRidge: 0.012 },
      { y: -0.22, rx: 0.036, rz: 0.045, backRidge: 0.012 }
    ], 16),
    materials.points
  );
  upperCannon.castShadow = true;
  lowerPivot.add(upperCannon);

  const lowerCannon = new THREE.Mesh(
    createLoftedLimb([
      { y: -0.20, rx: 0.037, rz: 0.046, backRidge: 0.012 },
      { y: -0.34, rx: 0.040, rz: 0.049, backRidge: 0.01 },
      { y: -0.38, rx: 0.045, rz: 0.054 }
    ], 16),
    cannonMat
  );
  lowerCannon.castShadow = true;
  lowerPivot.add(lowerCannon);

  const fetlockPivot = new THREE.Group();
  fetlockPivot.position.set(0, -0.38, 0);
  lowerPivot.add(fetlockPivot);

  const fetlockBulb = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 10), cannonMat);
  fetlockBulb.scale.set(0.94, 0.96, 1.12);
  fetlockBulb.position.set(0, 0, -0.008);
  fetlockBulb.castShadow = true;
  fetlockPivot.add(fetlockBulb);

  const pastern = new THREE.Mesh(
    createLoftedLimb([
      { y: 0.0, rx: 0.040, rz: 0.046 },
      { y: -0.12, rx: 0.036, rz: 0.042, z: 0.012 },
      { y: -0.22, rx: 0.042, rz: 0.050, z: 0.022 }
    ], 14),
    cannonMat
  );
  pastern.castShadow = true;
  fetlockPivot.add(pastern);

  const hoofPivot = buildHoofGroup(materials);
  hoofPivot.position.set(0, -0.22, 0.022);
  fetlockPivot.add(hoofPivot);

  return {
    upperPivot,
    lowerPivot,
    fetlockPivot,
    hoofPivot,
    sockMeshes: [lowerCannon, fetlockBulb, pastern]
  };
}

function buildHindLeg(side, hasSock, materials) {
  const upperPivot = new THREE.Group();
  upperPivot.position.set(side * 0.20, 0.04, -0.56);

  const thighMesh = new THREE.Mesh(
    createLoftedLimb([
      { y: 0.19, rx: 0.025, rz: 0.04, x: -side * 0.07, z: -0.02 },
      { y: 0.11, rx: 0.128, rz: 0.205, x: -side * 0.03, z: -0.02 },
      { y: -0.02, rx: 0.128, rz: 0.188, z: 0.01 },
      { y: -0.15, rx: 0.112, rz: 0.162, z: 0.02 },
      { y: -0.29, rx: 0.090, rz: 0.132, z: 0.0 },
      { y: -0.41, rx: 0.068, rz: 0.100, z: -0.085 },
      { y: -0.52, rx: 0.055, rz: 0.082, z: -0.15 }
    ], 18),
    materials.coat
  );
  thighMesh.castShadow = true;
  thighMesh.receiveShadow = true;
  upperPivot.add(thighMesh);

  const lowerPivot = new THREE.Group();
  lowerPivot.position.set(0, -0.52, -0.15);
  upperPivot.add(lowerPivot);

  const hockJoint = new THREE.Mesh(new THREE.SphereGeometry(0.055, 14, 12), materials.points);
  hockJoint.scale.set(0.88, 1.15, 1.24);
  hockJoint.position.set(0, 0.01, -0.014);
  hockJoint.castShadow = true;
  lowerPivot.add(hockJoint);

  const cannonMat = hasSock ? materials.whiteMarking : materials.points;
  const upperCannon = new THREE.Mesh(
    createLoftedLimb([
      { y: 0.0, rx: 0.049, rz: 0.066, z: 0.0 },
      { y: -0.15, rx: 0.039, rz: 0.050, z: 0.02, backRidge: 0.014 },
      { y: -0.24, rx: 0.037, rz: 0.047, z: 0.03, backRidge: 0.014 }
    ], 16),
    materials.points
  );
  upperCannon.castShadow = true;
  lowerPivot.add(upperCannon);

  const lowerCannon = new THREE.Mesh(
    createLoftedLimb([
      { y: -0.22, rx: 0.038, rz: 0.047, z: 0.028, backRidge: 0.014 },
      { y: -0.36, rx: 0.040, rz: 0.050, z: 0.04, backRidge: 0.01 },
      { y: -0.40, rx: 0.045, rz: 0.055, z: 0.045 }
    ], 16),
    cannonMat
  );
  lowerCannon.castShadow = true;
  lowerPivot.add(lowerCannon);

  const fetlockPivot = new THREE.Group();
  fetlockPivot.position.set(0, -0.40, 0.045);
  lowerPivot.add(fetlockPivot);

  const fetlockBulb = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 10), cannonMat);
  fetlockBulb.scale.set(0.94, 0.96, 1.12);
  fetlockBulb.position.set(0, 0, -0.008);
  fetlockBulb.castShadow = true;
  fetlockPivot.add(fetlockBulb);

  const pastern = new THREE.Mesh(
    createLoftedLimb([
      { y: 0.0, rx: 0.040, rz: 0.046 },
      { y: -0.12, rx: 0.036, rz: 0.042, z: 0.014 },
      { y: -0.22, rx: 0.042, rz: 0.050, z: 0.024 }
    ], 14),
    cannonMat
  );
  pastern.castShadow = true;
  fetlockPivot.add(pastern);

  const hoofPivot = buildHoofGroup(materials);
  hoofPivot.position.set(0, -0.22, 0.024);
  fetlockPivot.add(hoofPivot);

  return {
    upperPivot,
    lowerPivot,
    fetlockPivot,
    hoofPivot,
    sockMeshes: [lowerCannon, fetlockBulb, pastern]
  };
}

function buildManeLocks(lowerNeck, upperNeck, hairMaterial) {
  const locks = [];
  const lowerCount = 13;
  for (let i = 0; i < lowerCount; i += 1) {
    const t = i / (lowerCount - 1);
    const lockPivot = new THREE.Group();
    lockPivot.position.set(0.015, 0.22 + t * 0.30, -0.12 + t * 0.46);
    const len = 0.36 + Math.sin(t * Math.PI) * 0.18;
    const sideDrape = i % 3 === 0 ? -0.34 : 0.38;
    const mesh = new THREE.Mesh(createHairRibbonGeometry(0.14, len, 0.08, 6), hairMaterial);
    mesh.rotation.y = Math.PI * 0.5;
    mesh.rotation.z = sideDrape;
    mesh.castShadow = true;
    lockPivot.add(mesh);
    lowerNeck.add(lockPivot);
    locks.push({ pivot: lockPivot, phase: i * 0.38, baseZ: sideDrape * 0.5 });
  }

  const upperCount = 10;
  for (let i = 0; i < upperCount; i += 1) {
    const t = i / (upperCount - 1);
    const lockPivot = new THREE.Group();
    lockPivot.position.set(0.012, 0.15 + t * 0.25, -0.04 + t * 0.36);
    const len = 0.32 + (1 - t) * 0.14;
    const sideDrape = i % 3 === 0 ? -0.32 : 0.36;
    const mesh = new THREE.Mesh(createHairRibbonGeometry(0.125, len, 0.07, 6), hairMaterial);
    mesh.rotation.y = Math.PI * 0.5;
    mesh.rotation.z = sideDrape;
    mesh.castShadow = true;
    lockPivot.add(mesh);
    upperNeck.add(lockPivot);
    locks.push({ pivot: lockPivot, phase: (lowerCount + i) * 0.38, baseZ: sideDrape * 0.5 });
  }
  return locks;
}

function buildTailChain(parent, hairMaterial) {
  const segments = [];
  let current = new THREE.Group();
  current.position.set(0, 0.14, -0.90);
  parent.add(current);

  const segSpecs = [
    { len: 0.22, w: 0.13 },
    { len: 0.25, w: 0.17 },
    { len: 0.28, w: 0.20 },
    { len: 0.30, w: 0.19 },
    { len: 0.28, w: 0.15 }
  ];

  for (let i = 0; i < segSpecs.length; i += 1) {
    const spec = segSpecs[i];
    const segGroup = i === 0 ? current : new THREE.Group();
    if (i > 0) {
      segGroup.position.set(0, -segSpecs[i - 1].len * 0.85, -0.02);
      current.add(segGroup);
    }

    for (const angleY of [-0.62, 0, 0.62]) {
      const sheet = new THREE.Mesh(
        createHairRibbonGeometry(spec.w, spec.len * 1.55, 0.05, 6),
        hairMaterial
      );
      sheet.rotation.y = angleY;
      sheet.castShadow = true;
      segGroup.add(sheet);
    }
    segments.push(segGroup);
    current = segGroup;
  }

  return segments;
}

export function createHorseRig(coatId = 'bay') {
  const coat = COATS[coatId] || COATS.bay;
  const coatTex = createCoatTextures(coat);
  const hairTex = createHairTexture(coat.mane);

  const materials = {
    coat: new THREE.MeshPhysicalMaterial({
      map: coatTex.map,
      bumpMap: coatTex.bumpMap,
      bumpScale: 0.008,
      roughness: 0.48,
      metalness: 0.02,
      sheen: coat.sheen,
      sheenRoughness: 0.52,
      sheenColor: new THREE.Color(coat.bodyHighlight),
      clearcoat: 0.14,
      clearcoatRoughness: 0.55
    }),
    points: new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(coat.points),
      bumpMap: coatTex.bumpMap,
      bumpScale: 0.006,
      roughness: 0.58,
      metalness: 0.02,
      sheen: 0.28,
      sheenColor: new THREE.Color(coat.bodyPrimary)
    }),
    muzzle: new THREE.MeshStandardMaterial({
      color: new THREE.Color(coat.muzzle),
      roughness: 0.72,
      metalness: 0.0
    }),
    hoof: new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(coat.hoof),
      roughness: 0.34,
      metalness: 0.06,
      clearcoat: 0.38,
      clearcoatRoughness: 0.35
    }),
    whiteMarking: new THREE.MeshStandardMaterial({
      color: new THREE.Color('#eee9df'),
      roughness: 0.66
    }),
    eye: new THREE.MeshPhysicalMaterial({
      color: new THREE.Color('#140d09'),
      roughness: 0.05,
      metalness: 0.05,
      clearcoat: 1.0,
      clearcoatRoughness: 0.04
    }),
    darkCavity: new THREE.MeshStandardMaterial({
      color: new THREE.Color('#1a110f'),
      roughness: 0.92
    }),
    hair: new THREE.MeshStandardMaterial({
      map: hairTex,
      alphaTest: 0.2,
      transparent: true,
      side: THREE.DoubleSide,
      roughness: 0.62,
      metalness: 0.04
    })
  };

  const root = new THREE.Group();
  const bodyPivot = new THREE.Group();
  bodyPivot.position.set(0, 1.24, 0);
  root.add(bodyPivot);

  const thoraxGroup = new THREE.Group();
  bodyPivot.add(thoraxGroup);
  const thoraxMesh = buildThoraxMesh(materials.coat);
  thoraxGroup.add(thoraxMesh);

  const hindPivot = new THREE.Group();
  hindPivot.position.set(0, 0, -0.02);
  bodyPivot.add(hindPivot);
  const hindMesh = buildHindquartersMesh(materials.coat);
  hindPivot.add(hindMesh);

  const neckBasePivot = new THREE.Group();
  neckBasePivot.position.set(0, 0.20, 0.66);
  thoraxGroup.add(neckBasePivot);
  neckBasePivot.add(buildLowerNeckMesh(materials.coat));

  const neckMidPivot = new THREE.Group();
  neckMidPivot.position.set(0, 0.28, 0.33);
  neckBasePivot.add(neckMidPivot);
  neckMidPivot.add(buildUpperNeckMesh(materials.coat));

  const headPivot = new THREE.Group();
  headPivot.position.set(0, 0.26, 0.31);
  neckMidPivot.add(headPivot);
  const headParts = buildHeadGroup(materials);
  headParts.blaze.visible = coat.blaze;
  headPivot.add(headParts.head);

  const legHindLeft = buildHindLeg(1, coat.socks[0], materials);
  const legHindRight = buildHindLeg(-1, coat.socks[1], materials);
  const legForeLeft = buildForeLeg(1, coat.socks[2], materials);
  const legForeRight = buildForeLeg(-1, coat.socks[3], materials);

  hindPivot.add(legHindLeft.upperPivot, legHindRight.upperPivot);
  thoraxGroup.add(legForeLeft.upperPivot, legForeRight.upperPivot);

  const maneLocks = buildManeLocks(neckBasePivot, neckMidPivot, materials.hair);
  const tailSegments = buildTailChain(hindPivot, materials.hair);

  const legs = [legHindLeft, legHindRight, legForeLeft, legForeRight];
  const hoofTipLocal = new THREE.Vector3(0, -0.13, 0.026);
  const hoofWorldPositions = [
    new THREE.Vector3(),
    new THREE.Vector3(),
    new THREE.Vector3(),
    new THREE.Vector3()
  ];

  function applyCoat(nextCoatId) {
    const next = COATS[nextCoatId] || COATS.bay;
    const nextCoatTex = createCoatTextures(next);
    const nextHairTex = createHairTexture(next.mane);

    materials.coat.map?.dispose();
    materials.coat.bumpMap?.dispose();
    materials.hair.map?.dispose();

    materials.coat.map = nextCoatTex.map;
    materials.coat.bumpMap = nextCoatTex.bumpMap;
    materials.coat.sheen = next.sheen;
    materials.coat.sheenColor.set(next.bodyHighlight);
    materials.coat.needsUpdate = true;

    materials.points.color.set(next.points);
    materials.points.bumpMap = nextCoatTex.bumpMap;
    materials.points.sheenColor.set(next.bodyPrimary);
    materials.points.needsUpdate = true;

    materials.muzzle.color.set(next.muzzle);
    materials.hoof.color.set(next.hoof);
    materials.hair.map = nextHairTex;
    materials.hair.needsUpdate = true;

    headParts.blaze.visible = next.blaze;
    for (let i = 0; i < legs.length; i += 1) {
      const sockMat = next.socks[i] ? materials.whiteMarking : materials.points;
      for (const m of legs[i].sockMeshes) {
        m.material = sockMat;
      }
    }
  }

  function applyPose(pose, timeSec, speedFactor = 1) {
    bodyPivot.position.y = 1.24 + pose.bodyY;
    bodyPivot.rotation.x = pose.bodyPitch;
    bodyPivot.rotation.z = pose.bodyRoll;

    thoraxMesh.scale.set(pose.ribBreath, 1, 1);
    thoraxGroup.rotation.x = -pose.spineFlex * 0.35;
    hindPivot.rotation.x = pose.spineFlex * 0.55;

    neckBasePivot.rotation.x = pose.neckBasePitch;
    neckMidPivot.rotation.x = pose.neckMidPitch;
    headPivot.rotation.x = pose.headPitch;

    headParts.earLeft.rotation.x = 0.32 + pose.earLeftPitch;
    headParts.earRight.rotation.x = 0.32 + pose.earRightPitch;
    headParts.forelock.rotation.x = -0.85 - 0.18 * speedFactor - Math.sin(timeSec * 14) * 0.08;

    for (let i = 0; i < legs.length; i += 1) {
      const lp = pose.legs[i];
      const rig = legs[i];
      rig.upperPivot.rotation.x = lp.upper;
      rig.lowerPivot.rotation.x = lp.lower;
      rig.fetlockPivot.rotation.x = lp.fetlock;
      rig.hoofPivot.rotation.x = lp.hoof;
    }

    for (const lock of maneLocks) {
      const flutter = Math.sin(timeSec * (11 + 5 * speedFactor) - lock.phase) * (0.14 + 0.14 * speedFactor);
      lock.pivot.rotation.x = 0.42 * speedFactor + flutter * 0.55;
      lock.pivot.rotation.z = lock.baseZ + Math.cos(timeSec * 8 - lock.phase) * 0.1;
    }

    for (let i = 0; i < tailSegments.length; i += 1) {
      const seg = tailSegments[i];
      const lag = i * 0.55;
      const windLift = i === 0 ? pose.tailLift + 0.48 * speedFactor : 0.12 * speedFactor;
      const waveX = Math.sin(timeSec * 10 - lag) * (0.07 + 0.07 * speedFactor) + pose.tailWave * 0.32;
      const waveZ = Math.cos(timeSec * 7.5 - lag) * (0.05 + 0.05 * speedFactor);
      seg.rotation.x = windLift + waveX;
      seg.rotation.z = waveZ;
    }

    root.updateMatrixWorld(true);
    for (let i = 0; i < legs.length; i += 1) {
      hoofWorldPositions[i].copy(hoofTipLocal);
      legs[i].hoofPivot.localToWorld(hoofWorldPositions[i]);
    }
  }

  return {
    root,
    bodyPivot,
    legs,
    hoofWorldPositions,
    applyCoat,
    applyPose
  };
}
