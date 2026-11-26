import * as THREE from '/vendor/three/build/three.module.js';
import { renderPageCanvas, createDeskWoodTexture } from './paper-canvas.js';
import { hitTestArticle } from './editions.js';

const PAGE_WIDTH = 3.2;
const PAGE_HEIGHT = 4.4;
const SEG_X = 34;
const SEG_Y = 24;
const LEAF_COUNT = 4;

export class NewspaperScene {
  constructor(container, callbacks) {
    this.container = container;
    this.callbacks = callbacks;
    this.edition = null;
    this.spread = 0;
    this.folded = false;
    this.foldAmount = 0;
    this.loupeActive = false;
    this.warmLamp = true;
    this.pressBounce = 0;

    this.leaves = [];
    this.pageCanvases = [];
    this.pageTextures = [];

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2(99, 99);
    this.isDragging = false;
    this.dragMoved = false;
    this.dragStart = { x: 0, y: 0 };

    this.orbit = {
      yaw: 0,
      pitch: 1.08,
      radius: 6.2,
      targetYaw: 0,
      targetPitch: 1.08,
      targetRadius: 6.2,
      focus: new THREE.Vector3(0, 0.1, 0),
      targetFocus: new THREE.Vector3(0, 0.1, 0)
    };

    this.hoveredInfo = null;
    this.cornerHover = { right: 0, left: 0 };
    this.clock = new THREE.Clock();

    this.initRenderer();
    this.initLighting();
    this.initDeskAndProps();
    this.initDustMotes();
    this.initNewspaperRig();
    this.initHoverFrame();
    this.initLoupe();
    this.bindEvents();
    this.animate();
  }

  initRenderer() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#130d09');
    this.scene.fog = new THREE.FogExp2('#130d09', 0.042);

    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 60);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    this.container.appendChild(this.renderer.domElement);
  }

  initLighting() {
    this.ambientLight = new THREE.AmbientLight('#ffe9cc', 0.52);
    this.scene.add(this.ambientLight);

    this.keyLight = new THREE.DirectionalLight('#ffdfb0', 1.45);
    this.keyLight.position.set(-2.8, 8.2, 3.6);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.camera.near = 1;
    this.keyLight.shadow.camera.far = 22;
    this.keyLight.shadow.camera.left = -7;
    this.keyLight.shadow.camera.right = 7;
    this.keyLight.shadow.camera.top = 7;
    this.keyLight.shadow.camera.bottom = -7;
    this.keyLight.shadow.bias = -0.0005;
    this.scene.add(this.keyLight);

    this.lampLight = new THREE.PointLight('#ffb963', 2.8, 16, 1.4);
    this.lampLight.position.set(-4.3, 2.9, -2.1);
    this.lampLight.castShadow = true;
    this.scene.add(this.lampLight);

    this.rimLight = new THREE.DirectionalLight('#84a6c4', 0.42);
    this.rimLight.position.set(5.2, 4.0, -5.2);
    this.scene.add(this.rimLight);
  }

  initDeskAndProps() {
    const woodTex = new THREE.CanvasTexture(createDeskWoodTexture());
    woodTex.colorSpace = THREE.SRGBColorSpace;
    woodTex.wrapS = THREE.RepeatWrapping;
    woodTex.wrapT = THREE.RepeatWrapping;
    woodTex.repeat.set(2, 2);

    const desk = new THREE.Mesh(
      new THREE.PlaneGeometry(28, 20),
      new THREE.MeshStandardMaterial({
        map: woodTex,
        roughness: 0.62,
        metalness: 0.08
      })
    );
    desk.rotation.x = -Math.PI / 2;
    desk.position.y = -0.04;
    desk.receiveShadow = true;
    this.scene.add(desk);

    const brassMat = new THREE.MeshStandardMaterial({
      color: '#c99c48',
      roughness: 0.28,
      metalness: 0.88
    });
    const emeraldMat = new THREE.MeshStandardMaterial({
      color: '#125c3c',
      emissive: '#114d31',
      emissiveIntensity: 0.65,
      roughness: 0.22,
      metalness: 0.12
    });
    this.emeraldShadeMat = emeraldMat;

    const lampGroup = new THREE.Group();
    lampGroup.position.set(-4.8, 0, -2.5);
    lampGroup.rotation.y = 0.55;

    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.72, 0.16, 32), brassMat);
    base.position.y = 0.04;
    base.castShadow = true;
    lampGroup.add(base);

    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 2.75, 16), brassMat);
    stem.position.y = 1.4;
    stem.castShadow = true;
    lampGroup.add(stem);

    const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 1.45, 24, 1, false, 0, Math.PI), emeraldMat);
    shade.rotation.z = Math.PI / 2;
    shade.rotation.x = 0.45;
    shade.position.set(0.32, 2.72, 0.2);
    shade.castShadow = true;
    lampGroup.add(shade);

    const bulb = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 16, 16),
      new THREE.MeshBasicMaterial({ color: '#ffdf9e' })
    );
    bulb.position.set(0.32, 2.58, 0.22);
    this.lampBulb = bulb;
    lampGroup.add(bulb);
    this.scene.add(lampGroup);

    const penGroup = new THREE.Group();
    penGroup.position.set(4.55, 0.03, -1.65);
    penGroup.rotation.y = -0.42;
    const penBody = new THREE.Mesh(
      new THREE.CylinderGeometry(0.055, 0.048, 1.85, 16),
      new THREE.MeshStandardMaterial({ color: '#1b1614', roughness: 0.25, metalness: 0.2 })
    );
    penBody.rotation.z = Math.PI / 2;
    penBody.castShadow = true;
    penGroup.add(penBody);
    const penRing = new THREE.Mesh(new THREE.CylinderGeometry(0.062, 0.062, 0.09, 16), brassMat);
    penRing.rotation.z = Math.PI / 2;
    penRing.position.x = -0.18;
    penGroup.add(penRing);
    const penNib = new THREE.Mesh(new THREE.ConeGeometry(0.048, 0.28, 12), brassMat);
    penNib.rotation.z = Math.PI / 2;
    penNib.position.x = -1.05;
    penGroup.add(penNib);
    this.scene.add(penGroup);

    const weight = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.42, 0.22, 28), brassMat);
    weight.position.set(4.6, 0.07, -2.7);
    weight.castShadow = true;
    weight.receiveShadow = true;
    this.scene.add(weight);

    const inkwell = new THREE.Mesh(
      new THREE.CylinderGeometry(0.24, 0.30, 0.42, 20),
      new THREE.MeshStandardMaterial({ color: '#192229', roughness: 0.18, metalness: 0.25 })
    );
    inkwell.position.set(3.85, 0.17, -2.85);
    inkwell.castShadow = true;
    this.scene.add(inkwell);
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.12, 18), brassMat);
    cap.position.set(3.85, 0.44, -2.85);
    this.scene.add(cap);
  }

  initDustMotes() {
    const count = 110;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 12;
      positions[i * 3 + 1] = 0.4 + Math.random() * 4.5;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 9;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: '#ffe0a8',
      size: 0.035,
      transparent: true,
      opacity: 0.38,
      depthWrite: false
    });
    this.dust = new THREE.Points(geo, mat);
    this.scene.add(this.dust);
  }

  initNewspaperRig() {
    this.paperRig = new THREE.Group();
    this.scene.add(this.paperRig);

    for (let i = 0; i < LEAF_COUNT; i++) {
      const frontGeo = new THREE.PlaneGeometry(PAGE_WIDTH, PAGE_HEIGHT, SEG_X, SEG_Y);
      const backGeo = new THREE.PlaneGeometry(PAGE_WIDTH, PAGE_HEIGHT, SEG_X, SEG_Y);
      frontGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0.4, 0), 5.8);
      backGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0.4, 0), 5.8);

      const frontMat = new THREE.MeshStandardMaterial({
        color: '#ffffff',
        roughness: 0.88,
        metalness: 0.02,
        side: THREE.FrontSide
      });
      const backMat = new THREE.MeshStandardMaterial({
        color: '#ffffff',
        roughness: 0.88,
        metalness: 0.02,
        side: THREE.FrontSide
      });

      const frontMesh = new THREE.Mesh(frontGeo, frontMat);
      frontMesh.castShadow = true;
      frontMesh.receiveShadow = true;
      frontMesh.userData = { leafIndex: i, side: 'front', pageIndex: i * 2 };

      const backMesh = new THREE.Mesh(backGeo, backMat);
      backMesh.castShadow = true;
      backMesh.receiveShadow = true;
      backMesh.userData = { leafIndex: i, side: 'back', pageIndex: i * 2 + 1 };

      this.paperRig.add(frontMesh);
      this.paperRig.add(backMesh);

      this.leaves.push({
        index: i,
        progress: 0,
        targetProgress: 0,
        frontMesh,
        backMesh,
        frontGeo,
        backGeo
      });
    }
  }

  initHoverFrame() {
    const geo = new THREE.PlaneGeometry(1, 1);
    const mat = new THREE.MeshBasicMaterial({
      color: '#d49a38',
      transparent: true,
      opacity: 0.18,
      depthWrite: false
    });
    this.hoverHighlight = new THREE.Mesh(geo, mat);
    this.hoverHighlight.rotation.x = -Math.PI / 2;
    this.hoverHighlight.visible = false;
    this.paperRig.add(this.hoverHighlight);
  }

  initLoupe() {
    const group = new THREE.Group();
    const brassMat = new THREE.MeshStandardMaterial({
      color: '#d6a64c',
      roughness: 0.24,
      metalness: 0.9
    });
    const rim = new THREE.Mesh(new THREE.TorusGeometry(0.56, 0.055, 16, 48), brassMat);
    rim.rotation.x = Math.PI / 2;
    rim.castShadow = true;
    group.add(rim);

    const glass = new THREE.Mesh(
      new THREE.CylinderGeometry(0.54, 0.54, 0.03, 36),
      new THREE.MeshPhysicalMaterial({
        color: '#ffffff',
        transparent: true,
        opacity: 0.16,
        roughness: 0.06,
        metalness: 0.05
      })
    );
    group.add(glass);

    const handle = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.07, 1.05, 16),
      new THREE.MeshStandardMaterial({ color: '#3b2114', roughness: 0.42, metalness: 0.15 })
    );
    handle.rotation.z = Math.PI / 2;
    handle.rotation.y = -0.68;
    handle.position.set(0.84, -0.02, 0.68);
    handle.castShadow = true;
    group.add(handle);

    group.visible = false;
    this.loupeGroup = group;
    this.scene.add(group);
  }

  setEdition(edition) {
    this.edition = edition;
    this.pageTextures.forEach((tex) => tex.dispose());
    this.pageCanvases = [];
    this.pageTextures = [];

    for (let i = 0; i < 8; i++) {
      const canvas = renderPageCanvas(edition, i);
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
      this.pageCanvases.push(canvas);
      this.pageTextures.push(texture);
    }

    for (let i = 0; i < LEAF_COUNT; i++) {
      this.leaves[i].frontMesh.material.map = this.pageTextures[i * 2];
      this.leaves[i].frontMesh.material.needsUpdate = true;
      this.leaves[i].backMesh.material.map = this.pageTextures[i * 2 + 1];
      this.leaves[i].backMesh.material.needsUpdate = true;
    }
  }

  triggerPressStamp() {
    this.pressBounce = 1.0;
  }

  setSpread(nextSpread) {
    const clamped = Math.max(0, Math.min(LEAF_COUNT, nextSpread));
    if (clamped === this.spread) return false;
    this.spread = clamped;
    for (let i = 0; i < LEAF_COUNT; i++) {
      this.leaves[i].targetProgress = i < this.spread ? 1 : 0;
    }
    this.hoverHighlight.visible = false;
    this.updateDefaultCameraForSpread();
    return true;
  }

  updateDefaultCameraForSpread() {
    if (this.spread === 0 || this.spread === LEAF_COUNT) {
      this.orbit.targetRadius = this.folded ? 5.6 : 6.4;
      this.orbit.targetFocus.set(0, 0.1, 0);
    } else {
      this.orbit.targetRadius = this.folded ? 6.8 : 7.8;
      this.orbit.targetFocus.set(0, 0.1, 0);
    }
  }

  setFolded(flag) {
    this.folded = Boolean(flag);
    this.updateDefaultCameraForSpread();
  }

  setLoupe(flag) {
    this.loupeActive = Boolean(flag);
    if (!this.loupeActive) {
      this.loupeGroup.visible = false;
    }
  }

  setWarmLamp(flag) {
    this.warmLamp = Boolean(flag);
    if (this.warmLamp) {
      this.keyLight.color.set('#ffdfb0');
      this.keyLight.intensity = 1.45;
      this.lampLight.intensity = 2.8;
      this.ambientLight.color.set('#ffe9cc');
      this.ambientLight.intensity = 0.52;
      this.emeraldShadeMat.emissiveIntensity = 0.65;
      this.lampBulb.material.color.set('#ffdf9e');
      this.scene.background.set('#130d09');
      this.scene.fog.color.set('#130d09');
    } else {
      this.keyLight.color.set('#f5f7fc');
      this.keyLight.intensity = 1.85;
      this.lampLight.intensity = 0.45;
      this.ambientLight.color.set('#d9e3f0');
      this.ambientLight.intensity = 0.72;
      this.emeraldShadeMat.emissiveIntensity = 0.1;
      this.lampBulb.material.color.set('#b8b8a8');
      this.scene.background.set('#1d2026');
      this.scene.fog.color.set('#1d2026');
    }
  }

  resetCamera() {
    this.orbit.targetYaw = 0;
    this.orbit.targetPitch = 1.06;
    this.updateDefaultCameraForSpread();
  }

  focusOnArticle(pageIndex, article) {
    const isLeft = pageIndex % 2 === 1;
    const bx = article.box.x + article.box.w / 2;
    const by = article.box.y + article.box.h / 2;
    const localX = isLeft ? -PAGE_WIDTH * (1 - bx) : PAGE_WIDTH * bx;
    const localZ = (by - 0.5) * PAGE_HEIGHT;
    const worldX = localX + this.paperRig.position.x;
    this.orbit.targetFocus.set(worldX, 0.14, localZ);
    this.orbit.targetRadius = 4.2;
    this.orbit.targetPitch = 1.24;
  }

  updateLeafGeometry(leaf, time) {
    const p = leaf.progress;
    const angle = p * Math.PI;
    const sinFlip = Math.sin(p * Math.PI);
    const fold = this.foldAmount;

    const restRightY = (LEAF_COUNT - 1 - leaf.index) * 0.014;
    const restLeftY = leaf.index * 0.014;
    const baseY = restRightY * (1 - p) + restLeftY * p;

    const isTopRight = leaf.index === this.spread && p < 0.05;
    const isTopLeft = leaf.index === this.spread - 1 && p > 0.95;
    const cornerLift = isTopRight ? this.cornerHover.right : isTopLeft ? this.cornerHover.left : 0;

    const updatePlane = (geo, isBack) => {
      const pos = geo.attributes.position;
      const offsetY = isBack ? -0.0025 : 0.0025;

      for (let iy = 0; iy <= SEG_Y; iy++) {
        const vNorm = iy / SEG_Y;
        const zBase = (vNorm - 0.5) * PAGE_HEIGHT;

        for (let ix = 0; ix <= SEG_X; ix++) {
          const idx = iy * (SEG_X + 1) + ix;
          const uRaw = ix / SEG_X;
          const u = isBack ? 1 - uRaw : uRaw;

          const curlAngle = angle - sinFlip * 0.44 * (u * u);
          const dist = u * PAGE_WIDTH;

          let x = Math.cos(curlAngle) * dist;
          let y = baseY + Math.sin(curlAngle) * dist;

          y += sinFlip * 0.28 * Math.sin(u * Math.PI);

          const settleCurve = Math.sin(u * Math.PI) * 0.045 * (1 - sinFlip);
          const breathe = Math.sin(time * 1.4 + u * 3.1 + leaf.index) * 0.006 * u;
          y += settleCurve + breathe;

          if (cornerLift > 0.001) {
            const cornerWeight = Math.pow(u, 2.2) * Math.pow(Math.max(0, vNorm - 0.35) / 0.65, 1.8);
            y += cornerLift * cornerWeight * 0.42;
          }

          let z = zBase;
          if (fold > 0.001) {
            const zShift = fold * 0.92;
            y += fold * 0.22;
            if (vNorm > 0.5) {
              const dv = (vNorm - 0.5) * PAGE_HEIGHT;
              const foldAngle = fold * 2.95;
              z = zShift + Math.cos(foldAngle) * dv;
              y -= Math.sin(foldAngle) * dv * 0.12 + fold * 0.05 * (vNorm - 0.5);
            } else {
              z = zBase + zShift;
              y += fold * 0.09 * (0.5 - vNorm);
            }
          }

          const nx = -Math.sin(curlAngle);
          const ny = Math.cos(curlAngle);
          x += nx * offsetY;
          y += ny * offsetY;

          pos.setXYZ(idx, x, y, z);
        }
      }
      pos.needsUpdate = true;
      geo.computeVertexNormals();
    };

    updatePlane(leaf.frontGeo, false);
    updatePlane(leaf.backGeo, true);
  }

  getActiveMeshes() {
    const active = [];
    if (this.spread < LEAF_COUNT) {
      active.push(this.leaves[this.spread].frontMesh);
    }
    if (this.spread > 0) {
      active.push(this.leaves[this.spread - 1].backMesh);
    }
    return active;
  }

  pickPaperHit() {
    if (!this.edition) return null;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.getActiveMeshes(), false);
    if (!hits.length) return null;
    const hit = hits[0];
    const { pageIndex, side } = hit.object.userData;
    const uv = hit.uv;
    if (!uv) return null;
    const u = uv.x;
    const v = 1 - uv.y;
    const page = this.edition.pages[pageIndex];
    const article = hitTestArticle(page, u, v);
    return { hit, pageIndex, side, u, v, page, article };
  }

  updateHoverState() {
    const picked = this.pickPaperHit();
    this.hoveredInfo = picked;

    if (!picked) {
      this.hoverHighlight.visible = false;
      this.cornerHover.right *= 0.85;
      this.cornerHover.left *= 0.85;
      this.loupeGroup.visible = false;
      this.renderer.domElement.style.cursor = this.isDragging ? 'grabbing' : 'grab';
      this.callbacks.onHover?.(null);
      return;
    }

    const isRightCorner = picked.side === 'front' && picked.u > 0.78 && picked.v > 0.65;
    const isLeftCorner = picked.side === 'back' && picked.u < 0.22 && picked.v > 0.65;
    this.cornerHover.right += ((isRightCorner ? 1 : 0) - this.cornerHover.right) * 0.18;
    this.cornerHover.left += ((isLeftCorner ? 1 : 0) - this.cornerHover.left) * 0.18;

    if (this.loupeActive) {
      this.loupeGroup.visible = true;
      this.loupeGroup.position.lerp(
        new THREE.Vector3(picked.hit.point.x, picked.hit.point.y + 0.42, picked.hit.point.z),
        0.26
      );
      this.callbacks.onLoupeMove?.({
        canvas: this.pageCanvases[picked.pageIndex],
        u: picked.u,
        v: picked.v,
        pageNumber: picked.page.number
      });
    } else {
      this.loupeGroup.visible = false;
    }

    if (picked.article && !this.folded) {
      const b = picked.article.box;
      const isLeft = picked.pageIndex % 2 === 1;
      const localX = isLeft ? -PAGE_WIDTH * (1 - (b.x + b.w / 2)) : PAGE_WIDTH * (b.x + b.w / 2);
      const localZ = (b.y + b.h / 2 - 0.5) * PAGE_HEIGHT;
      this.hoverHighlight.scale.set(b.w * PAGE_WIDTH, b.h * PAGE_HEIGHT, 1);
      this.hoverHighlight.position.set(localX, 0.11, localZ);
      this.hoverHighlight.visible = true;
      this.renderer.domElement.style.cursor = 'pointer';
    } else {
      this.hoverHighlight.visible = false;
      this.renderer.domElement.style.cursor = isRightCorner || isLeftCorner ? 'pointer' : 'grab';
    }

    this.callbacks.onHover?.({
      pageNumber: picked.page.number,
      section: picked.page.section,
      article: picked.article,
      corner: isRightCorner ? 'next' : isLeftCorner ? 'prev' : null
    });
  }

  bindEvents() {
    const dom = this.renderer.domElement;

    dom.addEventListener('pointerdown', (e) => {
      const rect = dom.getBoundingClientRect();
      this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      this.isDragging = true;
      this.dragMoved = false;
      this.dragStart.x = e.clientX;
      this.dragStart.y = e.clientY;
    });

    window.addEventListener('pointermove', (e) => {
      const rect = dom.getBoundingClientRect();
      this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      if (this.isDragging) {
        const dx = e.clientX - this.dragStart.x;
        const dy = e.clientY - this.dragStart.y;
        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) {
          this.dragMoved = true;
        }
        this.orbit.targetYaw -= dx * 0.0065;
        this.orbit.targetYaw = Math.max(-1.1, Math.min(1.1, this.orbit.targetYaw));
        this.orbit.targetPitch += dy * 0.0055;
        this.orbit.targetPitch = Math.max(0.45, Math.min(1.48, this.orbit.targetPitch));
        this.dragStart.x = e.clientX;
        this.dragStart.y = e.clientY;
      }
    });

    window.addEventListener('pointerup', (e) => {
      if (!this.isDragging) return;
      this.isDragging = false;
      if (this.dragMoved) return;
      const rect = dom.getBoundingClientRect();
      if (
        e.clientX < rect.left ||
        e.clientX > rect.right ||
        e.clientY < rect.top ||
        e.clientY > rect.bottom
      ) {
        return;
      }
      const picked = this.pickPaperHit();
      if (!picked) return;

      if (picked.side === 'front' && picked.u > 0.85 && this.spread < LEAF_COUNT) {
        this.callbacks.onRequestSpread?.(this.spread + 1);
        return;
      }
      if (picked.side === 'back' && picked.u < 0.15 && this.spread > 0) {
        this.callbacks.onRequestSpread?.(this.spread - 1);
        return;
      }
      if (picked.article) {
        this.focusOnArticle(picked.pageIndex, picked.article);
        this.callbacks.onArticleSelect?.(picked.page, picked.article);
      }
    });

    dom.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.orbit.targetRadius += e.deltaY * 0.004;
        this.orbit.targetRadius = Math.max(3.2, Math.min(11.5, this.orbit.targetRadius));
      },
      { passive: false }
    );

    dom.addEventListener('dblclick', () => {
      this.resetCamera();
    });

    window.addEventListener('resize', () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    const dt = Math.min(0.05, this.clock.getDelta());
    const elapsed = this.clock.elapsedTime;

    const targetCenterX =
      this.spread === 0 ? -PAGE_WIDTH * 0.5 : this.spread === LEAF_COUNT ? PAGE_WIDTH * 0.5 : 0;
    this.paperRig.position.x += (targetCenterX - this.paperRig.position.x) * Math.min(1, dt * 6.5);

    if (this.pressBounce > 0.001) {
      this.pressBounce = Math.max(0, this.pressBounce - dt * 3.8);
      this.paperRig.position.y = Math.sin(this.pressBounce * Math.PI) * 0.16;
    } else {
      this.paperRig.position.y = 0;
    }

    const targetFold = this.folded ? 1 : 0;
    this.foldAmount += (targetFold - this.foldAmount) * Math.min(1, dt * 6.5);

    for (const leaf of this.leaves) {
      const diff = leaf.targetProgress - leaf.progress;
      if (Math.abs(diff) > 0.0005) {
        leaf.progress += diff * Math.min(1, dt * 6.2);
      } else {
        leaf.progress = leaf.targetProgress;
      }
      this.updateLeafGeometry(leaf, elapsed);
    }

    this.orbit.yaw += (this.orbit.targetYaw - this.orbit.yaw) * Math.min(1, dt * 7);
    this.orbit.pitch += (this.orbit.targetPitch - this.orbit.pitch) * Math.min(1, dt * 7);
    this.orbit.radius += (this.orbit.targetRadius - this.orbit.radius) * Math.min(1, dt * 7);
    this.orbit.focus.lerp(this.orbit.targetFocus, Math.min(1, dt * 6.5));

    const camX = this.orbit.focus.x + Math.sin(this.orbit.yaw) * Math.cos(this.orbit.pitch) * this.orbit.radius;
    const camY = this.orbit.focus.y + Math.sin(this.orbit.pitch) * this.orbit.radius;
    const camZ = this.orbit.focus.z + Math.cos(this.orbit.yaw) * Math.cos(this.orbit.pitch) * this.orbit.radius;
    this.camera.position.set(camX, camY, camZ);
    this.camera.lookAt(this.orbit.focus);

    if (this.dust) {
      this.dust.rotation.y = elapsed * 0.025;
      this.dust.position.y = Math.sin(elapsed * 0.45) * 0.12;
    }

    if (!this.isDragging) {
      this.updateHoverState();
    }

    this.renderer.render(this.scene, this.camera);
  }
}
