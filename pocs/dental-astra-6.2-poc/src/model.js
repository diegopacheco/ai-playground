import * as T from '../node_modules/three/build/three.module.js'
import { teeth } from './teeth.js'

const ivory = new T.MeshStandardMaterial({ color: 0xfff3d9, roughness: 0.32 })
const accent = new T.MeshStandardMaterial({ color: 0x78cbb4, roughness: 0.35 })
const tissue = new T.MeshStandardMaterial({ color: 0xd89489, roughness: 0.53 })
const angles = [0.105, 0.32, 0.53, 0.74, 0.95, 1.2, 1.48, 1.77]
const widths = [0.34, 0.31, 0.34, 0.36, 0.37, 0.46, 0.45, 0.41]
function center(t, angle) {
  const scale = t.upper ? 1 : 0.94
  return new T.Vector3(
    (t.right ? -1 : 1) * 2.62 * Math.sin(angle) * scale,
    0,
    (4.05 * Math.cos(angle) - 1.62) * scale,
  )
}
function gridMesh(rows, material) {
  const points = [],
    indices = [],
    n = rows[0].length
  for (let j = 0; j < rows.length; j++)
    for (let i = 0; i < n; i++) {
      points.push(...rows[j][i])
      if (j) {
        const a = (j - 1) * n + i,
          b = (j - 1) * n + ((i + 1) % n)
        indices.push(a, a + n, b, b, a + n, b + n)
      }
    }
  const geometry = new T.BufferGeometry()
  geometry.setAttribute('position', new T.Float32BufferAttribute(points, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  material.side = T.DoubleSide
  return new T.Mesh(geometry, material)
}
function crown(t) {
  const p = t.position,
    w = widths[p],
    d = p < 2 ? 0.23 : p === 2 ? 0.31 : p < 5 ? 0.4 : 0.49
  const rings = [
    [0.04, -0.32],
    [0.67, -0.25],
    [0.84, -0.12],
    [0.96, 0.04],
    [1, 0.19],
    [0.99, 0.29],
    [0.94, 0.35],
    [0.8, 0.38],
    [0.62, 0.39],
    [0.42, 0.4],
    [0.21, 0.4],
    [0.001, 0.4],
  ]
  const rows = rings.map(([r, h], j) =>
    Array.from({ length: 48 }, (_, i) => {
      const a = (-i * Math.PI) / 24
      const shape = (v) => Math.sign(v) * Math.abs(v) ** (p === 2 ? 0.93 : 0.69)
      const x = r * shape(Math.cos(a)),
        z = r * shape(Math.sin(a))
      let y = h
      if (j > 4) {
        const blend = Math.min(1, (j - 4) / 3)
        if (p < 2) y += blend * (0.22 * (1 - Math.abs(z)) - 0.06 * x * x)
        else if (p === 2) y += blend * (0.36 * (1 - Math.sqrt(x * x + z * z)))
        else {
          let cusps = 0
          for (const cz of [-0.5, 0.5])
            for (const cx of p < 5 ? [0] : [-0.5, 0.5])
              cusps += Math.exp(-((x - cx) ** 2 * 6 + (z - cz) ** 2 * 8))
          y += blend * (cusps * 0.19 - 0.06)
        }
      }
      return [x * w, y * (t.upper ? 1 : 0.9), z * d]
    }),
  )
  return gridMesh(rows, ivory)
}
function gum(upper) {
  const rows = Array.from({ length: 105 }, (_, i) => {
    const a = -1.91 + (i * 3.82) / 104,
      c = center({ right: false, upper }, a)
    const normal = new T.Vector3(
      4.05 * Math.sin(a),
      0,
      2.62 * Math.cos(a),
    ).normalize()
    const taper = Math.min(
      1,
      0.04 + Math.sin((Math.min(i, 104 - i, 4) * Math.PI) / 8),
    )
    return Array.from({ length: 32 }, (_, j) => {
      const b = (j * Math.PI) / 16,
        v = Math.cos(b) * (0.36 + 0.12 * Math.abs(Math.sin(a))) * taper
      return [
        c.x + normal.x * v,
        -0.34 + Math.sin(b) * 0.33 * taper + 0.034 * Math.cos(a * 29),
        c.z + normal.z * v,
      ]
    })
  })
  return gridMesh(rows, tissue)
}
export function mount(stage, select) {
  const canvas = stage.querySelector('canvas'),
    overlay = document.querySelector('#labels')
  const context = canvas.getContext('webgl2', { antialias: true, alpha: true })
  if (!context) throw new Error('WebGL 2 unavailable')
  const renderer = new T.WebGLRenderer({
    canvas,
    context,
    antialias: true,
    alpha: true,
  })
  renderer.toneMapping = T.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.96
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
  const scene = new T.Scene(),
    root = new T.Group(),
    camera = new T.OrthographicCamera(-5, 5, 5, -5, 0.1, 60)
  scene.add(root, new T.HemisphereLight(0xfffcf1, 0xe3c8b8, 2.25))
  for (const [x, y, z, intensity] of [
    [-4, 7, 8, 2.1],
    [5, -2, 4, 0.7],
  ]) {
    const light = new T.DirectionalLight(0xfff7e8, intensity)
    light.position.set(x, y, z)
    scene.add(light)
  }
  camera.position.z = 18
  const arches = [new T.Group(), new T.Group()],
    items = []
  function label(cls, content) {
    const el = document.createElement('span')
    el.className = cls
    el.textContent = content
    overlay.append(el)
    return el
  }
  for (const [index, g] of arches.entries()) {
    const upper = index === 0
    root.add(g)
    g.add(gum(upper))
    for (const t of teeth.filter((t) => t.upper === upper)) {
      const a = angles[t.position],
        mesh = crown(t)
      mesh.position.copy(center(t, a))
      mesh.rotation.y =
        (t.right ? -1 : 1) * Math.atan2(4.05 * Math.sin(a), 2.62 * Math.cos(a))
      mesh.userData.id = t.id
      g.add(mesh)
      items.push({ t, mesh, tag: label('tag', t.id) })
    }
  }
  const sides = [label('side', 'Patient R'), label('side', 'Patient L')]
  let arch = 'Both',
    id = 8,
    yaw = 0.12,
    pitch = 0,
    zoom = 1,
    frame = 0
  function schedule() {
    if (!frame) frame = requestAnimationFrame(draw)
  }
  function project(point) {
    point.project(camera)
    return {
      x: ((point.x + 1) * canvas.clientWidth) / 2,
      y: ((1 - point.y) * canvas.clientHeight) / 2,
    }
  }
  function place(el, p) {
    el.style.left = p.x + 'px'
    el.style.top = p.y + 'px'
  }
  function draw() {
    frame = 0
    const w = canvas.clientWidth,
      h = canvas.clientHeight
    renderer.setSize(w, h, false)
    const both = arch === 'Both',
      height = Math.max(both ? 8.6 : 6.3, ((both ? 8.15 : 8.4) * h) / w) / zoom
    Object.assign(camera, {
      top: height / 2,
      bottom: -height / 2,
      left: (-height * w) / h / 2,
      right: (height * w) / h / 2,
    })
    camera.updateProjectionMatrix()
    root.rotation.set(pitch, yaw, 0, 'YXZ')
    for (const [i, g] of arches.entries()) {
      const sign = i === 0 ? -1 : 1
      g.visible = both || arch === (i === 0 ? 'Upper' : 'Lower')
      g.scale.y = sign
      g.rotation.x = sign * (both ? 0.49 : 1.12)
      g.position.y = both ? -sign * 1.99 : 0
    }
    scene.updateMatrixWorld(true)
    camera.updateMatrixWorld()
    const boxes = []
    for (const { t, mesh, tag } of [...items].sort(
      (a, b) => Number(b.t.id === id) - Number(a.t.id === id),
    )) {
      const selected = t.id === id
      mesh.material = selected ? accent : ivory
      tag.classList.toggle('active', selected)
      const p = project(mesh.localToWorld(new T.Vector3(0, -0.52, 0.15)))
      const fits = p.x > 12 && p.x < w - 12 && p.y > 12 && p.y < h - 12
      const show =
        mesh.parent.visible &&
        fits &&
        !boxes.some((q) => Math.abs(q.x - p.x) < 24 && Math.abs(q.y - p.y) < 19)
      tag.hidden = !show
      if (show) {
        place(tag, p)
        boxes.push(p)
      }
    }
    sides.forEach((side, i) => {
      const p = project(
        root.localToWorld(new T.Vector3((i ? 1 : -1) * 3.2, 0, 0)),
      )
      place(side, p)
      side.hidden =
        p.x < 26 ||
        p.x > w - 26 ||
        boxes.some((q) => Math.abs(q.x - p.x) < 40 && Math.abs(q.y - p.y) < 17)
    })
    renderer.render(scene, camera)
  }
  const ray = new T.Raycaster()
  let drag
  canvas.onpointerdown = (e) => {
    drag = { x: e.clientX, y: e.clientY, yaw, pitch, moved: false }
    canvas.setPointerCapture(e.pointerId)
  }
  canvas.onpointermove = (e) => {
    if (!drag) return
    const dx = e.clientX - drag.x,
      dy = e.clientY - drag.y
    if (Math.hypot(dx, dy) > 3) drag.moved = true
    if (drag.moved) {
      yaw = drag.yaw + dx * 0.009
      pitch = T.MathUtils.clamp(drag.pitch + dy * 0.008, -1.55, 1.55)
      schedule()
    }
  }
  canvas.onpointercancel = () => {
    drag = null
  }
  canvas.onpointerup = (e) => {
    if (drag && !drag.moved) {
      const r = canvas.getBoundingClientRect()
      ray.setFromCamera(
        new T.Vector2(
          (2 * (e.clientX - r.left)) / r.width - 1,
          1 - (2 * (e.clientY - r.top)) / r.height,
        ),
        camera,
      )
      const hit = ray.intersectObjects(
        arches.filter((g) => g.visible).flatMap((g) => g.children),
        false,
      )[0]
      if (hit?.object.userData.id) select(hit.object.userData.id)
    }
    drag = null
  }
  function magnify(f) {
    zoom = T.MathUtils.clamp(zoom * f, 0.7, 2.2)
    schedule()
  }
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault()
      magnify(Math.exp(-e.deltaY * 0.0015))
    },
    { passive: false },
  )
  document.querySelector('#plus').onclick = () => magnify(1.15)
  document.querySelector('#minus').onclick = () => magnify(1 / 1.15)
  function reset() {
    yaw = 0.12
    pitch = 0
    zoom = 1
    schedule()
  }
  document.querySelector('#reset').onclick = reset
  new ResizeObserver(schedule).observe(stage)
  return {
    set(nextId, nextArch) {
      if (arch !== nextArch) reset()
      id = nextId
      arch = nextArch
      schedule()
    },
  }
}
