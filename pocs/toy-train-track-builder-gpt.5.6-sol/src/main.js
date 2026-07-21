import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js'
import './style.css'

const canvas = document.querySelector('#scene')
const viewport = document.querySelector('#viewport')
let fallbackCanvas = null
let fallbackContext = null
let fallbackActive = false

const createFallbackRenderer = () => {
  fallbackActive = true
  fallbackCanvas = document.createElement('canvas')
  fallbackCanvas.className = 'fallback-scene'
  fallbackCanvas.setAttribute('aria-hidden', 'true')
  canvas.insertAdjacentElement('afterend', fallbackCanvas)
  fallbackContext = fallbackCanvas.getContext('2d')
  let ratio = 1
  return {
    shadowMap: {},
    outputColorSpace: '',
    toneMapping: 0,
    toneMappingExposure: 1,
    setPixelRatio(value) { ratio = value },
    setSize(width, height) {
      fallbackCanvas.width = Math.round(width * ratio)
      fallbackCanvas.height = Math.round(height * ratio)
      fallbackCanvas.style.width = `${width}px`
      fallbackCanvas.style.height = `${height}px`
      fallbackContext.setTransform(ratio, 0, 0, ratio, 0, 0)
    },
    render() { drawFallbackScene() }
  }
}

const webglProbe = document.createElement('canvas')
let renderer
if (webglProbe.getContext('webgl2')) {
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false })
  } catch {
    renderer = createFallbackRenderer()
  }
} else renderer = createFallbackRenderer()
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFSoftShadowMap
renderer.outputColorSpace = THREE.SRGBColorSpace
renderer.toneMapping = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure = 1.08

const scene = new THREE.Scene()
scene.background = new THREE.Color('#d9eef1')
scene.fog = new THREE.Fog('#d9eef1', 24, 48)

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100)
camera.position.set(18, 17, 20)

const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true
controls.dampingFactor = 0.07
controls.target.set(0, 0, 0)
controls.minDistance = 12
controls.maxDistance = 34
controls.maxPolarAngle = Math.PI * 0.46
controls.minPolarAngle = Math.PI * 0.18

const colors = {
  wood: '#c99357',
  woodLight: '#e0b77c',
  woodDark: '#85552f',
  rail: '#69442b',
  red: '#c84e3d',
  redDark: '#8e332d',
  green: '#3d7658',
  yellow: '#efb841',
  blue: '#46799a',
  cream: '#fff3ce',
  ground: '#dfc68e',
  snow: '#eff5f1'
}

const mat = (color, roughness = 0.8, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness })
const materials = {
  wood: mat(colors.wood, 0.72),
  woodLight: mat(colors.woodLight, 0.78),
  woodDark: mat(colors.woodDark, 0.76),
  rail: mat(colors.rail, 0.66),
  red: mat(colors.red, 0.68),
  redDark: mat(colors.redDark, 0.7),
  green: mat(colors.green, 0.75),
  yellow: mat(colors.yellow, 0.7),
  blue: mat(colors.blue, 0.7),
  cream: mat(colors.cream, 0.82),
  black: mat('#29312e', 0.6),
  silver: mat('#a9b5ae', 0.35, 0.45),
  glass: new THREE.MeshStandardMaterial({ color: '#b9e4e9', roughness: 0.18, metalness: 0.05, transparent: true, opacity: 0.78 }),
  water: new THREE.MeshStandardMaterial({ color: '#69aeb7', roughness: 0.24, transparent: true, opacity: 0.88 }),
  leaf: mat('#527f4d', 0.9),
  leafDark: mat('#315b3e', 0.9),
  white: mat('#f5f4e8', 0.82)
}

const ambient = new THREE.HemisphereLight('#fff5d8', '#8d7757', 2.2)
scene.add(ambient)
const sun = new THREE.DirectionalLight('#fff1c9', 4.1)
sun.position.set(-9, 18, 8)
sun.castShadow = true
sun.shadow.mapSize.set(2048, 2048)
sun.shadow.camera.left = -18
sun.shadow.camera.right = 18
sun.shadow.camera.top = 14
sun.shadow.camera.bottom = -14
sun.shadow.bias = -0.0004
scene.add(sun)
const nightLamp = new THREE.PointLight('#f4bd63', 0, 24, 1.6)
nightLamp.position.set(0, 8, 2)
scene.add(nightLamp)

const table = new THREE.Group()
scene.add(table)
const tableBase = new THREE.Mesh(new RoundedBoxGeometry(28, 0.7, 21, 8, 0.5), mat('#b98751', 0.8))
tableBase.position.y = -0.52
tableBase.receiveShadow = true
tableBase.castShadow = true
table.add(tableBase)
const groundMaterial = mat(colors.ground, 0.92)
const ground = new THREE.Mesh(new RoundedBoxGeometry(27.3, 0.25, 20.3, 8, 0.42), groundMaterial)
ground.position.y = -0.08
ground.receiveShadow = true
table.add(ground)

const gridGroup = new THREE.Group()
table.add(gridGroup)
const gridMaterial = new THREE.LineBasicMaterial({ color: '#8b6f45', transparent: true, opacity: 0.16 })
const gridPoints = []
const cellSize = 2.15
for (let x = -5.5; x <= 5.5; x += 1) {
  gridPoints.push(new THREE.Vector3(x * cellSize, 0.065, -4.5 * cellSize), new THREE.Vector3(x * cellSize, 0.065, 4.5 * cellSize))
}
for (let z = -4.5; z <= 4.5; z += 1) {
  gridPoints.push(new THREE.Vector3(-5.5 * cellSize, 0.065, z * cellSize), new THREE.Vector3(5.5 * cellSize, 0.065, z * cellSize))
}
gridGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(gridPoints), gridMaterial))

const box = (w, h, d, material, x = 0, y = 0, z = 0, radius = 0.05) => {
  const mesh = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 3, Math.min(radius, w / 3, h / 3, d / 3)), material)
  mesh.position.set(x, y, z)
  mesh.castShadow = true
  mesh.receiveShadow = true
  return mesh
}

const cylinder = (radius, depth, material, x = 0, y = 0, z = 0, axis = 'y', sides = 20) => {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, depth, sides), material)
  mesh.position.set(x, y, z)
  if (axis === 'x') mesh.rotation.z = Math.PI / 2
  if (axis === 'z') mesh.rotation.x = Math.PI / 2
  mesh.castShadow = true
  mesh.receiveShadow = true
  return mesh
}

const addTree = (x, z, scale = 1) => {
  const tree = new THREE.Group()
  tree.add(cylinder(0.16 * scale, 1.2 * scale, materials.woodDark, 0, 0.62 * scale, 0))
  const crown1 = new THREE.Mesh(new THREE.ConeGeometry(0.83 * scale, 1.65 * scale, 10), materials.leaf)
  crown1.position.y = 1.45 * scale
  crown1.castShadow = true
  tree.add(crown1)
  const crown2 = new THREE.Mesh(new THREE.ConeGeometry(0.62 * scale, 1.25 * scale, 10), materials.leafDark)
  crown2.position.y = 2.15 * scale
  crown2.castShadow = true
  tree.add(crown2)
  tree.position.set(x, 0, z)
  scenery.add(tree)
}

const addHouse = (x, z, wallMaterial, roofMaterial, rotation = 0) => {
  const house = new THREE.Group()
  house.add(box(2.05, 1.55, 1.7, wallMaterial, 0, 0.8, 0, 0.12))
  const roof = new THREE.Mesh(new THREE.ConeGeometry(1.55, 1, 4), roofMaterial)
  roof.position.y = 1.95
  roof.rotation.y = Math.PI / 4
  roof.castShadow = true
  house.add(roof)
  house.add(box(0.56, 0.82, 0.06, materials.woodDark, 0, 0.46, 0.88, 0.03))
  house.add(box(0.4, 0.4, 0.06, materials.glass, -0.62, 0.96, 0.88, 0.03))
  house.add(box(0.4, 0.4, 0.06, materials.glass, 0.62, 0.96, 0.88, 0.03))
  house.position.set(x, 0, z)
  house.rotation.y = rotation
  scenery.add(house)
}

const scenery = new THREE.Group()
scene.add(scenery)
addTree(-11.6, -7.9, 1.1)
addTree(-9.7, -8.2, 0.82)
addTree(10.8, -7.6, 1.05)
addTree(11.8, -5.6, 0.76)
addTree(-11.7, 7.6, 0.88)
addTree(11.6, 7.2, 0.95)
addHouse(-10.5, 5.4, materials.cream, materials.red, 0.2)
addHouse(10.3, 5.1, materials.blue, materials.yellow, -0.25)

const pond = new THREE.Mesh(new THREE.CircleGeometry(2.15, 48), new THREE.MeshStandardMaterial({ color: '#6ba8a9', roughness: 0.25, metalness: 0.08, transparent: true, opacity: 0.82 }))
pond.rotation.x = -Math.PI / 2
pond.scale.set(1.4, 0.8, 1)
pond.position.set(9.8, 0.08, -5.1)
pond.receiveShadow = true
scenery.add(pond)
for (let i = 0; i < 7; i += 1) {
  const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(0.18 + (i % 3) * 0.05, 0), materials.woodLight)
  const angle = i / 7 * Math.PI * 2
  rock.position.set(9.8 + Math.cos(angle) * 2.6, 0.18, -5.1 + Math.sin(angle) * 1.72)
  rock.scale.y = 0.65
  rock.castShadow = true
  scenery.add(rock)
}

const cloudGroup = new THREE.Group()
scene.add(cloudGroup)
for (let i = 0; i < 4; i += 1) {
  const cloud = new THREE.Group()
  for (let j = 0; j < 5; j += 1) {
    const puff = new THREE.Mesh(new THREE.SphereGeometry(0.8 + (j % 2) * 0.25, 18, 12), materials.white)
    puff.scale.y = 0.65
    puff.position.set((j - 2) * 0.75, Math.abs(j - 2) * -0.12, (j % 2) * 0.35)
    cloud.add(puff)
  }
  cloud.position.set(-12 + i * 8, 10 + (i % 2) * 2, -8 + (i % 3) * 6)
  cloudGroup.add(cloud)
}
cloudGroup.visible = false

const starGroup = new THREE.Group()
scene.add(starGroup)
const starPositions = []
for (let i = 0; i < 260; i += 1) {
  const theta = Math.random() * Math.PI * 2
  const radius = 28 + Math.random() * 8
  starPositions.push(Math.cos(theta) * radius, 9 + Math.random() * 22, Math.sin(theta) * radius)
}
const starGeometry = new THREE.BufferGeometry()
starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3))
starGroup.add(new THREE.Points(starGeometry, new THREE.PointsMaterial({ color: '#f6e9bb', size: 0.16, sizeAttenuation: true })))
starGroup.visible = false

const snowGroup = new THREE.Group()
scene.add(snowGroup)
for (let i = 0; i < 420; i += 1) {
  const flake = new THREE.Mesh(new THREE.SphereGeometry(0.035 + Math.random() * 0.07, 5, 4), materials.white)
  flake.position.set((Math.random() - 0.5) * 26, 0.12 + Math.random() * 0.12, (Math.random() - 0.5) * 19)
  flake.scale.y = 0.4
  snowGroup.add(flake)
}
snowGroup.visible = false

const trackGroup = new THREE.Group()
scene.add(trackGroup)
const tracks = new Map()
const builderScenery = new THREE.Group()
scene.add(builderScenery)
const lakeConnections = new THREE.Group()
builderScenery.add(lakeConnections)
const sceneryItems = new Map()
let selectedPiece = 'straight'
let selectedRotation = 1
let selectedTrackKey = null
let eraseMode = false
let mode = 'build'
let history = []
const sceneryTypes = new Set(['tree', 'lake', 'house'])

const directions = [
  { x: 0, z: -1, name: 'N' },
  { x: 1, z: 0, name: 'E' },
  { x: 0, z: 1, name: 'S' },
  { x: -1, z: 0, name: 'W' }
]

const endpoints = piece => {
  let base = [0, 2]
  if (piece.type === 'curve') base = [0, 1]
  if (piece.type === 't-junction') base = [0, 1, 3]
  if (piece.type === 'x-crossing') base = [0, 1, 2, 3]
  return base.map(value => (value + piece.rotation) % 4)
}

const trackRise = 0.9
const endpointHeight = (piece, direction) => {
  const localDirection = (direction - piece.rotation + 4) % 4
  if (piece.type === 'track-over') return localDirection === 0 ? trackRise : 0
  if (piece.type === 'track-under') return localDirection === 2 ? trackRise : 0
  return 0
}

const createSleepers = group => {
  for (let i = -3; i <= 3; i += 1) group.add(box(1.36, 0.13, 0.22, materials.woodLight, 0, 0.13, i * 0.28, 0.04))
}

const createStraightRails = group => {
  group.add(box(0.1, 0.15, 2.02, materials.rail, -0.43, 0.24, 0, 0.03))
  group.add(box(0.1, 0.15, 2.02, materials.rail, 0.43, 0.24, 0, 0.03))
}

const createJunctionTrack = (group, type) => {
  const crossing = new THREE.Group()
  createSleepers(crossing)
  createStraightRails(crossing)
  crossing.rotation.y = Math.PI / 2
  group.add(crossing)
  if (type === 'x-crossing') {
    createSleepers(group)
    createStraightRails(group)
    return
  }
  for (let i = 0; i < 4; i += 1) group.add(box(1.36, 0.13, 0.22, materials.woodLight, 0, 0.13, -0.12 - i * 0.28, 0.04))
  group.add(box(0.1, 0.15, 1.04, materials.rail, -0.43, 0.24, -0.52, 0.03))
  group.add(box(0.1, 0.15, 1.04, materials.rail, 0.43, 0.24, -0.52, 0.03))
}

const createTrackSlope = (group, descending) => {
  for (const railX of [-0.43, 0.43]) {
    const points = []
    for (let i = 0; i <= 20; i += 1) {
      const t = i / 20
      const height = descending ? (1 - t) * trackRise : t * trackRise
      points.push(new THREE.Vector3(railX, 0.25 + height, -cellSize / 2 + t * cellSize))
    }
    const rail = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 28, 0.065, 6, false), materials.rail)
    rail.castShadow = true
    group.add(rail)
  }
  for (let i = 0; i <= 7; i += 1) {
    const t = i / 7
    const height = descending ? (1 - t) * trackRise : t * trackRise
    const sleeper = box(1.36, 0.13, 0.22, materials.woodLight, 0, 0.13 + height, -cellSize / 2 + t * cellSize, 0.04)
    sleeper.rotation.x = descending ? 0.4 : -0.4
    group.add(sleeper)
  }
  const supportZ = descending ? -0.72 : 0.72
  group.add(box(0.16, 0.78, 0.16, materials.red, -0.62, 0.45, supportZ, 0.03))
  group.add(box(0.16, 0.78, 0.16, materials.red, 0.62, 0.45, supportZ, 0.03))
}

const curvePath = new THREE.QuadraticBezierCurve3(
  new THREE.Vector3(0, 0, -cellSize / 2),
  new THREE.Vector3(cellSize / 2, 0, -cellSize / 2),
  new THREE.Vector3(cellSize / 2, 0, 0)
)

const createCurveTrack = group => {
  for (const offset of [-0.43, 0.43]) {
    const points = []
    for (let i = 0; i <= 24; i += 1) {
      const angle = -Math.PI + i / 24 * Math.PI / 2
      const radius = cellSize / 2 + offset
      points.push(new THREE.Vector3(cellSize / 2 + Math.cos(angle) * radius, 0.25, -cellSize / 2 - Math.sin(angle) * radius))
    }
    const rail = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 24, 0.065, 6, false), materials.rail)
    rail.castShadow = true
    group.add(rail)
  }
  for (let i = 1; i < 9; i += 1) {
    const t = i / 10
    const p = curvePath.getPoint(t)
    const tangent = curvePath.getTangent(t)
    const sleeper = box(1.36, 0.13, 0.2, materials.woodLight, p.x, 0.13, p.z, 0.04)
    sleeper.rotation.y = Math.atan2(tangent.x, tangent.z)
    group.add(sleeper)
  }
}

const addBridge = group => {
  for (const side of [-0.76, 0.76]) {
    group.add(box(0.11, 1.15, 2.0, materials.red, side, 0.72, 0, 0.03))
    for (const z of [-0.88, 0, 0.88]) group.add(box(0.14, 1.35, 0.14, materials.redDark, side, 0.66, z, 0.03))
    for (const z of [-0.45, 0.45]) {
      const brace = box(0.09, 1.15, 0.09, materials.red, side, 0.74, z, 0.02)
      brace.rotation.x = z > 0 ? -0.68 : 0.68
      group.add(brace)
    }
  }
}

const addStation = group => {
  group.add(box(0.75, 0.16, 1.92, materials.cream, 0.9, 0.16, 0, 0.05))
  group.add(box(0.88, 1.1, 1.25, materials.yellow, 1.1, 0.71, 0.2, 0.08))
  group.add(box(1.05, 0.14, 1.48, materials.red, 1.1, 1.31, 0.2, 0.04))
  group.add(box(0.07, 0.55, 0.48, materials.glass, 0.64, 0.85, 0.2, 0.02))
  group.add(box(0.07, 0.55, 0.48, materials.woodDark, 1.56, 0.57, 0.2, 0.02))
  group.add(box(0.08, 0.08, 1.15, materials.cream, 0.52, 1.01, 0.2, 0.02))
}

const addOverpass = group => {
  group.add(box(2.2, 0.22, 0.72, materials.silver, 0, 1.28, 0, 0.08))
  group.add(box(2.2, 0.08, 0.08, materials.cream, 0, 1.43, -0.31, 0.02))
  group.add(box(2.2, 0.08, 0.08, materials.cream, 0, 1.43, 0.31, 0.02))
  group.add(box(0.2, 1.12, 0.2, materials.red, -0.88, 0.65, 0, 0.04))
  group.add(box(0.2, 1.12, 0.2, materials.red, 0.88, 0.65, 0, 0.04))
}

const createSceneryMesh = item => {
  const group = new THREE.Group()
  if (item.type === 'tree') {
    group.add(cylinder(0.16, 1.05, materials.woodDark, 0, 0.54, 0))
    const lower = new THREE.Mesh(new THREE.ConeGeometry(0.8, 1.3, 10), materials.leafDark)
    lower.position.y = 1.25
    lower.castShadow = true
    group.add(lower)
    const upper = new THREE.Mesh(new THREE.ConeGeometry(0.58, 1.1, 10), materials.leaf)
    upper.position.y = 1.9
    upper.castShadow = true
    group.add(upper)
  }
  if (item.type === 'lake') {
    const water = new THREE.Mesh(new THREE.CircleGeometry(1.18, 36), materials.water)
    water.rotation.x = -Math.PI / 2
    water.scale.set(1, 0.82, 1)
    water.position.y = 0.08
    group.add(water)
    for (let i = 0; i < 8; i += 1) {
      const angle = i / 8 * Math.PI * 2
      const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(0.12 + i % 2 * 0.04, 0), materials.woodLight)
      rock.position.set(Math.cos(angle) * 0.98, 0.11, Math.sin(angle) * 0.7)
      rock.scale.y = 0.6
      rock.userData.lakeEdge = { dx: Math.round(Math.cos(angle)), dz: Math.round(Math.sin(angle)) }
      group.add(rock)
    }
  }
  if (item.type === 'house') {
    group.add(box(1.35, 1.1, 1.2, materials.yellow, 0, 0.58, 0, 0.1))
    const roof = new THREE.Mesh(new THREE.ConeGeometry(1.03, 0.68, 4), materials.red)
    roof.position.y = 1.46
    roof.rotation.y = Math.PI / 4
    roof.castShadow = true
    group.add(roof)
    group.add(box(0.35, 0.65, 0.06, materials.green, 0, 0.36, 0.63, 0.02))
    group.add(box(0.28, 0.28, 0.06, materials.glass, -0.42, 0.82, 0.63, 0.02))
    group.add(box(0.28, 0.28, 0.06, materials.glass, 0.42, 0.82, 0.63, 0.02))
  }
  group.position.set(item.x * cellSize, 0.08, item.z * cellSize)
  group.rotation.y = item.type === 'lake' ? 0 : -item.rotation * Math.PI / 2
  group.userData.key = `${item.x},${item.z}`
  builderScenery.add(group)
  item.mesh = group
  return group
}

function updateLakeConnections() {
  lakeConnections.clear()
  const offsets = [[1, 0], [0, 1], [1, 1], [1, -1]]
  for (const item of sceneryItems.values()) {
    if (item.type !== 'lake') continue
    for (const child of item.mesh.children) {
      if (!child.userData.lakeEdge) continue
      const { dx, dz } = child.userData.lakeEdge
      child.visible = sceneryItems.get(`${item.x + dx},${item.z + dz}`)?.type !== 'lake'
    }
    for (const [dx, dz] of offsets) {
      const neighbor = sceneryItems.get(`${item.x + dx},${item.z + dz}`)
      if (!neighbor || neighbor.type !== 'lake') continue
      const diagonal = dx !== 0 && dz !== 0
      const width = diagonal ? 1.5 : dx ? cellSize + 0.2 : 1.65
      const depth = diagonal ? 1.5 : dz ? cellSize + 0.2 : 1.65
      const connector = box(width, 0.045, depth, materials.water, (item.x + dx / 2) * cellSize, 0.13, (item.z + dz / 2) * cellSize, 0.22)
      connector.receiveShadow = true
      lakeConnections.add(connector)
    }
  }
}

const createTrackMesh = piece => {
  const group = new THREE.Group()
  if (piece.type === 'curve') createCurveTrack(group)
  else if (piece.type === 't-junction' || piece.type === 'x-crossing') createJunctionTrack(group, piece.type)
  else if (piece.type === 'track-over') createTrackSlope(group, true)
  else if (piece.type === 'track-under') createTrackSlope(group, false)
  else {
    createSleepers(group)
    createStraightRails(group)
  }
  if (piece.type === 'bridge') addBridge(group)
  if (piece.type === 'station') addStation(group)
  if (piece.type === 'overpass') addOverpass(group)
  group.position.set(piece.x * cellSize, 0.08 + (piece.elevation || 0), piece.z * cellSize)
  group.rotation.y = -piece.rotation * Math.PI / 2
  group.userData.key = `${piece.x},${piece.z}`
  group.userData.piece = piece
  trackGroup.add(group)
  piece.mesh = group
  return group
}

const updateTrackElevations = () => {
  const visited = new Set()
  for (const seed of tracks.values()) {
    const seedKey = `${seed.x},${seed.z}`
    if (visited.has(seedKey)) continue
    const component = []
    seed.elevation = 0
    visited.add(seedKey)
    const queue = [seed]
    while (queue.length) {
      const current = queue.shift()
      component.push(current)
      for (const direction of endpoints(current)) {
        const next = tracks.get(`${current.x + directions[direction].x},${current.z + directions[direction].z}`)
        const reverse = (direction + 2) % 4
        if (!next || !endpoints(next).includes(reverse)) continue
        const nextKey = `${next.x},${next.z}`
        if (visited.has(nextKey)) continue
        next.elevation = current.elevation + endpointHeight(current, direction) - endpointHeight(next, reverse)
        visited.add(nextKey)
        queue.push(next)
      }
    }
    let lowest = Infinity
    for (const piece of component) {
      for (const direction of endpoints(piece)) lowest = Math.min(lowest, piece.elevation + endpointHeight(piece, direction))
    }
    if (lowest < 0) for (const piece of component) piece.elevation -= lowest
  }
  for (const piece of tracks.values()) {
    if (piece.mesh) trackGroup.remove(piece.mesh)
    createTrackMesh(piece)
  }
}

const addTrack = (x, z, type, rotation, record = true) => {
  const key = `${x},${z}`
  const scenery = sceneryItems.get(key)
  const lakeTrack = scenery?.type === 'lake'
  if (tracks.has(key) || scenery && !lakeTrack || Math.abs(x) > 5 || Math.abs(z) > 4) return false
  if (record) saveHistory()
  const piece = { x, z, type, rotation: rotation % 4 }
  tracks.set(key, piece)
  updateTrackElevations()
  updatePieceCount()
  return true
}

const addSceneryItem = (x, z, type, rotation, record = true) => {
  const key = `${x},${z}`
  const track = tracks.get(key)
  const lakeTrack = type === 'lake' && Boolean(track)
  if (sceneryItems.has(key) || track && !lakeTrack || Math.abs(x) > 5 || Math.abs(z) > 4) return false
  if (record) saveHistory()
  const item = { x, z, type, rotation: rotation % 4 }
  sceneryItems.set(key, item)
  createSceneryMesh(item)
  updateLakeConnections()
  updatePieceCount()
  return true
}

const removeTrack = (key, record = true) => {
  const piece = tracks.get(key)
  if (!piece) return false
  if (record) saveHistory()
  trackGroup.remove(piece.mesh)
  tracks.delete(key)
  updateTrackElevations()
  if (selectedTrackKey === key) clearSelectedTrack()
  updatePieceCount()
  return true
}

const removeSceneryItem = (key, record = true) => {
  const item = sceneryItems.get(key)
  if (!item) return false
  if (record) saveHistory()
  builderScenery.remove(item.mesh)
  sceneryItems.delete(key)
  updateLakeConnections()
  if (selectedTrackKey === key) clearSelectedTrack()
  updatePieceCount()
  return true
}

const removePlacedItem = (key, record = true) => tracks.has(key) ? removeTrack(key, record) : removeSceneryItem(key, record)

const serializeTracks = () => [...tracks.values()].map(({ x, z, type, rotation }) => ({ x, z, type, rotation }))
const serializeScenery = () => [...sceneryItems.values()].map(({ x, z, type, rotation }) => ({ x, z, type, rotation }))
const saveHistory = () => {
  history.push({ tracks: serializeTracks(), scenery: serializeScenery() })
  if (history.length > 30) history.shift()
}
const restoreTracks = state => {
  clearSelectedTrack()
  for (const piece of tracks.values()) trackGroup.remove(piece.mesh)
  for (const item of sceneryItems.values()) builderScenery.remove(item.mesh)
  tracks.clear()
  sceneryItems.clear()
  for (const piece of state.tracks) addTrack(piece.x, piece.z, piece.type, piece.rotation, false)
  for (const item of state.scenery) addSceneryItem(item.x, item.z, item.type, item.rotation, false)
  updateLakeConnections()
  updatePieceCount()
}
const undo = () => {
  if (!history.length) return showToast('Nothing to undo yet')
  restoreTracks(history.pop())
  showToast('Last change undone')
}

const starterTracks = []
for (let x = -3; x <= 3; x += 1) {
  starterTracks.push({ x, z: -2, type: x === 0 ? 'station' : 'straight', rotation: 1 })
  starterTracks.push({ x, z: 2, type: x === 1 ? 'bridge' : 'straight', rotation: 1 })
}
for (let z = -1; z <= 1; z += 1) {
  starterTracks.push({ x: -4, z, type: 'straight', rotation: 0 })
  starterTracks.push({ x: 4, z, type: z === 0 ? 'bridge' : 'straight', rotation: 0 })
}
starterTracks.push({ x: -4, z: -2, type: 'curve', rotation: 1 })
starterTracks.push({ x: 4, z: -2, type: 'curve', rotation: 2 })
starterTracks.push({ x: 4, z: 2, type: 'curve', rotation: 3 })
starterTracks.push({ x: -4, z: 2, type: 'curve', rotation: 0 })
for (const piece of starterTracks) addTrack(piece.x, piece.z, piece.type, piece.rotation, false)

const highlighterMaterial = new THREE.MeshBasicMaterial({ color: '#4f9a69', transparent: true, opacity: 0.32, depthWrite: false })
const highlighter = new THREE.Mesh(new THREE.PlaneGeometry(cellSize * 0.9, cellSize * 0.9), highlighterMaterial)
highlighter.rotation.x = -Math.PI / 2
highlighter.position.y = 0.09
highlighter.visible = false
scene.add(highlighter)

const selectionMaterial = new THREE.MeshBasicMaterial({ color: '#f2bd43', transparent: true, opacity: 0.88, depthWrite: false, side: THREE.DoubleSide })
const selectionMarker = new THREE.Mesh(new THREE.RingGeometry(cellSize * 0.47, cellSize * 0.57, 32), selectionMaterial)
selectionMarker.rotation.x = -Math.PI / 2
selectionMarker.position.y = 0.12
selectionMarker.visible = false
scene.add(selectionMarker)

function clearSelectedTrack() {
  selectedTrackKey = null
  selectionMarker.visible = false
  document.querySelector('.build-tip').lastChild.textContent = ' Drag a piece onto the table or click an open square.'
}

const selectTrack = key => {
  const piece = tracks.get(key) || sceneryItems.get(key)
  if (!piece) return false
  selectedTrackKey = key
  selectionMarker.position.set(piece.x * cellSize, 0.12, piece.z * cellSize)
  selectionMarker.visible = mode === 'build'
  document.querySelector('.build-tip').lastChild.textContent = ' Track selected. Use Rotate or Erase.'
  return true
}

const setEraseMode = active => {
  eraseMode = active
  document.querySelector('#eraseButton').classList.toggle('active', eraseMode)
  viewport.classList.toggle('erase-mode', eraseMode)
}

const raycaster = new THREE.Raycaster()
const pointer = new THREE.Vector2()
let pointerStart = null
let hoverCell = null

const setPointer = event => {
  const rect = canvas.getBoundingClientRect()
  pointer.x = (event.clientX - rect.left) / rect.width * 2 - 1
  pointer.y = -(event.clientY - rect.top) / rect.height * 2 + 1
  raycaster.setFromCamera(pointer, camera)
}

const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
const canLayerAt = (key, type) => sceneryItems.get(key)?.type === 'lake' && !tracks.has(key) && !sceneryTypes.has(type) || type === 'lake' && tracks.has(key) && !sceneryItems.has(key)
const updateHover = event => {
  if (mode !== 'build') return
  setPointer(event)
  const point = new THREE.Vector3()
  if (!raycaster.ray.intersectPlane(groundPlane, point)) return
  const x = Math.round(point.x / cellSize)
  const z = Math.round(point.z / cellSize)
  const valid = Math.abs(x) <= 5 && Math.abs(z) <= 4
  hoverCell = valid ? { x, z } : null
  highlighter.visible = valid
  if (!valid) return
  highlighter.position.set(x * cellSize, 0.1, z * cellSize)
  const occupied = tracks.has(`${x},${z}`) || sceneryItems.has(`${x},${z}`)
  const canLayer = canLayerAt(`${x},${z}`, selectedPiece)
  highlighterMaterial.color.set(eraseMode ? occupied ? '#c74f3d' : '#997f63' : occupied && !canLayer ? '#f2bd43' : '#4f9a69')
}

const placeAtHover = () => {
  if (!hoverCell) return false
  const key = `${hoverCell.x},${hoverCell.z}`
  if (eraseMode) {
    if (removePlacedItem(key)) {
      showToast('Track piece returned to the box')
      return true
    }
    return false
  }
  if ((tracks.has(key) || sceneryItems.has(key)) && !canLayerAt(key, selectedPiece)) {
    selectTrack(key)
    showToast('Track selected')
    return true
  }
  const added = sceneryTypes.has(selectedPiece)
    ? addSceneryItem(hoverCell.x, hoverCell.z, selectedPiece, selectedRotation)
    : addTrack(hoverCell.x, hoverCell.z, selectedPiece, selectedRotation)
  if (added) {
    selectTrack(key)
    showToast(`${selectedPiece[0].toUpperCase()}${selectedPiece.slice(1)} piece placed`)
    return true
  }
  return false
}

canvas.addEventListener('pointerdown', event => { pointerStart = { x: event.clientX, y: event.clientY } })
canvas.addEventListener('pointermove', updateHover)
canvas.addEventListener('pointerleave', () => {
  highlighter.visible = false
  hoverCell = null
})
canvas.addEventListener('pointerup', event => {
  if (!pointerStart || mode !== 'build') return
  const distance = Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y)
  pointerStart = null
  if (distance > 6 || !hoverCell) return
  placeAtHover()
})

const createWheel = (x, z) => {
  const wheel = cylinder(0.25, 0.14, materials.black, x, 0.25, z, 'x', 18)
  const hub = cylinder(0.1, 0.16, materials.yellow, x, 0.25, z, 'x', 14)
  return [wheel, hub]
}

const createEngine = () => {
  const train = new THREE.Group()
  train.add(box(1.1, 0.55, 1.45, materials.red, 0, 0.57, 0.04, 0.16))
  train.add(cylinder(0.4, 0.95, materials.red, 0, 0.88, -0.45, 'z', 24))
  train.add(box(0.92, 0.92, 0.62, materials.redDark, 0, 1.03, 0.5, 0.1))
  train.add(box(0.64, 0.5, 0.07, materials.glass, 0, 1.1, 0.18, 0.02))
  train.add(cylinder(0.16, 0.52, materials.black, 0, 1.35, -0.62, 'y', 18))
  train.add(cylinder(0.27, 0.12, materials.black, 0, 1.64, -0.62, 'y', 18))
  for (const z of [-0.48, 0.5]) for (const part of createWheel(-0.56, z)) train.add(part)
  for (const z of [-0.48, 0.5]) for (const part of createWheel(0.56, z)) train.add(part)
  train.scale.setScalar(0.72)
  return train
}

const createWagon = (colorMaterial, cargo) => {
  const wagon = new THREE.Group()
  wagon.add(box(1.12, 0.52, 1.34, colorMaterial, 0, 0.56, 0, 0.12))
  wagon.add(box(1.2, 0.12, 1.45, materials.woodDark, 0, 0.28, 0, 0.04))
  if (cargo === 'food') {
    wagon.add(box(0.44, 0.38, 0.44, materials.cream, -0.28, 1, -0.25, 0.04))
    wagon.add(box(0.44, 0.38, 0.44, materials.woodLight, 0.28, 1, 0.25, 0.04))
    for (let i = 0; i < 5; i += 1) {
      const fruit = new THREE.Mesh(new THREE.SphereGeometry(0.11, 10, 8), i % 2 ? materials.red : materials.green)
      fruit.position.set(-0.35 + i * 0.17, 1.23, 0.08)
      wagon.add(fruit)
    }
  }
  if (cargo === 'minerals') {
    for (let i = 0; i < 8; i += 1) {
      const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(0.17 + i % 3 * 0.025, 0), i % 3 ? materials.silver : materials.black)
      rock.position.set(-0.38 + i % 3 * 0.38, 0.94 + i % 2 * 0.1, -0.38 + Math.floor(i / 3) * 0.35)
      rock.castShadow = true
      wagon.add(rock)
    }
  }
  if (cargo === 'people') {
    wagon.add(box(1.08, 0.68, 1.28, materials.green, 0, 0.98, 0, 0.08))
    for (const z of [-0.38, 0, 0.38]) {
      wagon.add(box(0.08, 0.3, 0.25, materials.glass, -0.56, 1.03, z, 0.02))
      wagon.add(box(0.08, 0.3, 0.25, materials.glass, 0.56, 1.03, z, 0.02))
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 8), materials.woodLight)
      head.position.set(0, 1.18, z)
      wagon.add(head)
    }
  }
  if (cargo === 'timber') {
    for (let i = 0; i < 5; i += 1) wagon.add(cylinder(0.13, 1.18, materials.woodLight, -0.3 + i % 3 * 0.3, 0.93 + Math.floor(i / 3) * 0.22, 0, 'z', 12))
  }
  if (cargo === 'milk') {
    for (const x of [-0.3, 0.3]) for (const z of [-0.35, 0.35]) wagon.add(cylinder(0.19, 0.55, materials.cream, x, 1, z, 'y', 16))
  }
  if (cargo === 'mail') {
    wagon.add(box(0.72, 0.56, 0.55, materials.cream, -0.16, 1, -0.22, 0.04))
    wagon.add(box(0.62, 0.44, 0.48, materials.yellow, 0.22, 0.94, 0.36, 0.04))
    wagon.add(box(0.75, 0.06, 0.06, materials.red, -0.16, 1.02, -0.51, 0.01))
  }
  if (cargo === 'animals') {
    for (const z of [-0.32, 0.32]) {
      wagon.add(box(0.45, 0.3, 0.25, materials.cream, 0, 0.98, z, 0.1))
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.16, 10, 8), materials.cream)
      head.position.set(0, 1.12, z - 0.18)
      wagon.add(head)
    }
  }
  if (cargo === 'tools') {
    wagon.add(box(0.82, 0.45, 0.75, materials.redDark, 0, 0.95, 0, 0.05))
    wagon.add(box(0.55, 0.08, 0.08, materials.silver, 0, 1.23, 0, 0.02))
    wagon.add(box(0.08, 0.25, 0.08, materials.silver, -0.22, 1.13, 0, 0.02))
  }
  if (cargo === 'caboose') {
    wagon.add(box(0.94, 0.72, 1.02, materials.red, 0, 0.99, 0, 0.08))
    wagon.add(box(1.08, 0.12, 1.18, materials.yellow, 0, 1.39, 0, 0.04))
    wagon.add(box(0.45, 0.34, 0.06, materials.glass, 0, 1.04, -0.53, 0.02))
  }
  for (const z of [-0.4, 0.4]) for (const part of createWheel(-0.56, z)) wagon.add(part)
  for (const z of [-0.4, 0.4]) for (const part of createWheel(0.56, z)) wagon.add(part)
  wagon.scale.setScalar(0.72)
  return wagon
}

const trainParts = [
  createEngine(),
  createWagon(materials.yellow, 'food'),
  createWagon(materials.blue, 'minerals'),
  createWagon(materials.green, 'people'),
  createWagon(materials.red, 'timber'),
  createWagon(materials.blue, 'milk'),
  createWagon(materials.green, 'mail'),
  createWagon(materials.yellow, 'animals'),
  createWagon(materials.green, 'tools'),
  createWagon(materials.redDark, 'caboose')
]
for (const part of trainParts) {
  part.visible = false
  scene.add(part)
}

const steamParticles = []
for (let i = 0; i < 22; i += 1) {
  const material = new THREE.MeshBasicMaterial({ color: '#f7f3e7', transparent: true, opacity: 0 })
  const particle = new THREE.Mesh(new THREE.SphereGeometry(0.22, 10, 8), material)
  particle.visible = false
  particle.userData.life = 0
  particle.userData.velocity = new THREE.Vector3()
  steamParticles.push(particle)
  scene.add(particle)
}
let steamAccumulator = 0

const updateSteam = delta => {
  steamAccumulator += trainRunning ? delta : 0
  if (trainRunning && trainParts[0].visible && steamAccumulator >= 0.11) {
    steamAccumulator = 0
    const particle = steamParticles.find(value => !value.visible)
    if (particle) {
      const chimney = trainParts[0].localToWorld(new THREE.Vector3(0, 1.68, -0.62))
      particle.position.copy(chimney)
      particle.scale.setScalar(0.7)
      particle.material.opacity = 0.86
      particle.visible = true
      particle.userData.life = 1
      particle.userData.velocity.set((Math.random() - 0.5) * 0.12, 0.72 + Math.random() * 0.24, (Math.random() - 0.5) * 0.12)
    }
  }
  for (const particle of steamParticles) {
    if (!particle.visible) continue
    particle.userData.life -= delta * 0.72
    particle.position.addScaledVector(particle.userData.velocity, delta)
    particle.position.x += Math.sin(particle.position.y * 5) * delta * 0.05
    particle.scale.addScalar(delta * 1.05)
    particle.material.opacity = Math.max(0, particle.userData.life) * 0.82
    if (particle.userData.life <= 0) particle.visible = false
  }
}

let route = { points: [], cumulative: [], length: 0, closed: false, count: 0 }
let trainRunning = false
let trainDistance = 0
let trainDirection = 1

const opposite = direction => (direction + 2) % 4
const neighborKey = (piece, direction) => `${piece.x + directions[direction].x},${piece.z + directions[direction].z}`

const connectedAt = (piece, direction) => {
  const next = tracks.get(neighborKey(piece, direction))
  return next && endpoints(next).includes(opposite(direction)) ? next : null
}

const routePointsForPiece = (piece, entering, forcedLeaving = null) => {
  const exits = endpoints(piece)
  const available = exits.filter(direction => direction !== entering && connectedAt(piece, direction))
  const straight = opposite(entering)
  const leaving = forcedLeaving ?? (available.includes(straight) ? straight : available[0] ?? exits.find(direction => direction !== entering))
  const center = new THREE.Vector3(piece.x * cellSize, 0.41 + (piece.elevation || 0), piece.z * cellSize)
  const pointFor = direction => new THREE.Vector3(
    piece.x * cellSize + directions[direction].x * cellSize / 2,
    0.41 + (piece.elevation || 0) + endpointHeight(piece, direction),
    piece.z * cellSize + directions[direction].z * cellSize / 2
  )
  const start = pointFor(entering)
  const end = pointFor(leaving)
  if (piece.type !== 'curve' && leaving === straight) {
    const values = []
    for (let i = 0; i <= 12; i += 1) {
      const t = i / 12
      const point = start.clone().lerp(end, t)
      values.push(point)
    }
    return { values, leaving }
  }
  const control = piece.type === 'curve'
    ? center.clone().add(new THREE.Vector3((directions[entering].x + directions[leaving].x) * cellSize / 2, (start.y + end.y) / 2 - center.y, (directions[entering].z + directions[leaving].z) * cellSize / 2))
    : center
  const curve = new THREE.QuadraticBezierCurve3(start, control, end)
  return { values: curve.getPoints(18), leaving }
}

const buildRoute = () => {
  const pieces = [...tracks.values()]
  if (!pieces.length) return { points: [], cumulative: [], length: 0, closed: false, count: 0 }
  const seen = new Set()
  let component = []
  for (const seed of pieces) {
    const seedKey = `${seed.x},${seed.z}`
    if (seen.has(seedKey)) continue
    const currentComponent = []
    const stack = [seed]
    seen.add(seedKey)
    while (stack.length) {
      const current = stack.pop()
      currentComponent.push(current)
      for (const direction of endpoints(current)) {
        const next = connectedAt(current, direction)
        const nextKey = next ? `${next.x},${next.z}` : ''
        if (next && !seen.has(nextKey)) {
          seen.add(nextKey)
          stack.push(next)
        }
      }
    }
    if (currentComponent.length > component.length) component = currentComponent
  }
  const degree = piece => endpoints(piece).filter(direction => connectedAt(piece, direction)).length
  const start = component.find(piece => degree(piece) < 2) || component[0]
  const startEndpoints = endpoints(start)
  let entering = startEndpoints.find(direction => !connectedAt(start, direction))
  if (entering === undefined) entering = startEndpoints[0]
  const values = []
  const visited = new Set()
  let current = start
  let currentEntering = entering
  let closed = false
  while (current && !visited.has(`${current.x},${current.z}`)) {
    const currentKey = `${current.x},${current.z}`
    visited.add(currentKey)
    const segment = routePointsForPiece(current, currentEntering)
    values.push(...segment.values.slice(values.length ? 1 : 0))
    const next = connectedAt(current, segment.leaving)
    if (!next) break
    if (next === start) {
      closed = true
      values.push(values[0].clone())
      break
    }
    current = next
    currentEntering = opposite(segment.leaving)
  }
  const cumulative = [0]
  for (let i = 1; i < values.length; i += 1) cumulative.push(cumulative[i - 1] + values[i].distanceTo(values[i - 1]))
  return { points: values, cumulative, length: cumulative.at(-1) || 0, closed, count: visited.size }
}

const buildSwitchingRoute = baseRoute => {
  const pieces = [...tracks.values()]
  if (!pieces.length) return baseRoute
  const degree = piece => endpoints(piece).filter(direction => connectedAt(piece, direction)).length
  const start = pieces.find(piece => degree(piece) === 1) || pieces.find(piece => piece.type !== 't-junction' && piece.type !== 'x-crossing' && degree(piece) > 1) || pieces[0]
  const startEndpoints = endpoints(start)
  let entering = startEndpoints.find(direction => !connectedAt(start, direction))
  if (entering === undefined) entering = startEndpoints[0]
  const values = []
  const visited = new Set()
  const junctionVisits = new Map()
  let current = start
  let currentEntering = entering
  let switchCount = 0
  for (let step = 0; step < 800 && current; step += 1) {
    const key = `${current.x},${current.z}`
    visited.add(key)
    const exits = endpoints(current).filter(direction => direction !== currentEntering && connectedAt(current, direction))
    if (!exits.length) {
      const leaving = endpoints(current).find(direction => direction !== currentEntering)
      if (leaving === undefined) break
      const segment = routePointsForPiece(current, currentEntering, leaving)
      values.push(...segment.values.slice(values.length ? 1 : 0))
      values.push(...segment.values.slice(0, -1).reverse())
      const previous = connectedAt(current, currentEntering)
      if (!previous) break
      current = previous
      currentEntering = opposite(currentEntering)
      continue
    }
    const straight = opposite(currentEntering)
    const ordered = exits.includes(straight) ? [straight, ...exits.filter(direction => direction !== straight)] : exits
    let leaving = ordered[0]
    if ((current.type === 't-junction' || current.type === 'x-crossing') && ordered.length > 1) {
      const visit = junctionVisits.get(key) || 0
      leaving = ordered[visit % ordered.length]
      junctionVisits.set(key, visit + 1)
      switchCount += 1
    }
    const segment = routePointsForPiece(current, currentEntering, leaving)
    values.push(...segment.values.slice(values.length ? 1 : 0))
    const next = connectedAt(current, leaving)
    if (!next) break
    current = next
    currentEntering = opposite(leaving)
  }
  if (!switchCount || values.length < 2) return baseRoute
  const cumulative = [0]
  for (let i = 1; i < values.length; i += 1) cumulative.push(cumulative[i - 1] + values[i].distanceTo(values[i - 1]))
  return {
    points: values,
    cumulative,
    length: cumulative.at(-1) || 0,
    closed: false,
    count: visited.size,
    switching: true,
    displayLength: baseRoute.length
  }
}

const sampleRoute = distance => {
  if (route.points.length < 2) return null
  let value = distance
  let direction = trainDirection
  if (route.closed) value = ((value % route.length) + route.length) % route.length
  else {
    const cycle = route.length * 2
    value = ((value % cycle) + cycle) % cycle
    if (value > route.length) {
      value = cycle - value
      direction *= -1
    }
  }
  let low = 0
  let high = route.cumulative.length - 1
  while (low < high) {
    const mid = Math.floor((low + high) / 2)
    if (route.cumulative[mid] < value) low = mid + 1
    else high = mid
  }
  const index = Math.max(1, low)
  const startDistance = route.cumulative[index - 1]
  const span = route.cumulative[index] - startDistance || 1
  const t = (value - startDistance) / span
  const position = route.points[index - 1].clone().lerp(route.points[index], t)
  const tangent = route.points[index].clone().sub(route.points[index - 1]).normalize().multiplyScalar(direction)
  return { position, tangent }
}

const positionTrainPart = (part, distance) => {
  const sample = sampleRoute(distance)
  if (!sample) return
  part.position.copy(sample.position)
  part.lookAt(sample.position.clone().sub(sample.tangent))
}

const prepareSimulation = () => {
  const baseRoute = buildRoute()
  route = buildSwitchingRoute(baseRoute)
  document.querySelector('#routeLength').textContent = `${(route.displayLength ?? route.length).toFixed(1)} m`
  const message = document.querySelector('#routeMessage')
  if (route.count < 2) message.textContent = 'Connect at least two matching track pieces.'
  else if (route.switching) message.textContent = `Automatic switches alternate routes across ${route.count} connected pieces.`
  else if (route.closed) message.textContent = `A closed loop with ${route.count} connected pieces is ready.`
  else message.textContent = `An open route with ${route.count} connected pieces is ready.`
  trainDistance = 0
  trainDirection = 1
  trainParts.forEach((part, index) => {
    part.visible = route.count >= 2
    positionTrainPart(part, -index * 1.15)
  })
  if (route.count < 2) setTrainRunning(false)
}

let audioContext = null
let audioMaster = null
let windGain = null
let rainGain = null
let steamGain = null
let trainOscillator = null
let trainGain = null
let muted = false
let nextNatureSound = 0
let nextPeopleSound = 0
const worldState = { sky: 'day', weather: 'sunny', ground: 'clean' }

const ensureAudio = () => {
  if (audioContext) {
    if (audioContext.state === 'suspended') audioContext.resume()
    return audioContext
  }
  const AudioEngine = window.AudioContext || window.webkitAudioContext
  if (!AudioEngine) return null
  audioContext = new AudioEngine()
  audioMaster = audioContext.createGain()
  audioMaster.gain.value = muted ? 0 : 0.7
  audioMaster.connect(audioContext.destination)
  const length = audioContext.sampleRate * 2
  const buffer = audioContext.createBuffer(1, length, audioContext.sampleRate)
  const values = buffer.getChannelData(0)
  for (let i = 0; i < length; i += 1) values[i] = Math.random() * 2 - 1
  const noise = audioContext.createBufferSource()
  noise.buffer = buffer
  noise.loop = true
  const windFilter = audioContext.createBiquadFilter()
  windFilter.type = 'lowpass'
  windFilter.frequency.value = 520
  windGain = audioContext.createGain()
  windGain.gain.value = 0
  const rainFilter = audioContext.createBiquadFilter()
  rainFilter.type = 'bandpass'
  rainFilter.frequency.value = 2400
  rainFilter.Q.value = 0.5
  rainGain = audioContext.createGain()
  rainGain.gain.value = 0
  const steamFilter = audioContext.createBiquadFilter()
  steamFilter.type = 'highpass'
  steamFilter.frequency.value = 900
  steamGain = audioContext.createGain()
  steamGain.gain.value = 0
  noise.connect(windFilter).connect(windGain).connect(audioMaster)
  noise.connect(rainFilter).connect(rainGain).connect(audioMaster)
  noise.connect(steamFilter).connect(steamGain).connect(audioMaster)
  noise.start()
  return audioContext
}

const playTone = (frequency, duration, volume, type = 'sine', endFrequency = frequency) => {
  if (muted || !ensureAudio()) return
  const oscillator = audioContext.createOscillator()
  const gain = audioContext.createGain()
  const now = audioContext.currentTime
  oscillator.type = type
  oscillator.frequency.setValueAtTime(frequency, now)
  oscillator.frequency.exponentialRampToValueAtTime(Math.max(20, endFrequency), now + duration)
  gain.gain.setValueAtTime(0.0001, now)
  gain.gain.exponentialRampToValueAtTime(volume, now + 0.02)
  gain.gain.exponentialRampToValueAtTime(0.0001, now + duration)
  oscillator.connect(gain).connect(audioMaster)
  oscillator.start(now)
  oscillator.stop(now + duration + 0.02)
}

const setMuted = value => {
  muted = value
  if (!muted) ensureAudio()
  if (audioMaster) audioMaster.gain.setTargetAtTime(muted ? 0 : 0.7, audioContext.currentTime, 0.03)
  const soundButton = document.querySelector('#soundButton')
  soundButton.classList.toggle('muted', muted)
  soundButton.setAttribute('aria-pressed', String(muted))
  soundButton.setAttribute('aria-label', muted ? 'Enable sound' : 'Mute sound')
  document.querySelectorAll('[data-sound]').forEach(button => button.classList.toggle('active', button.dataset.sound === (muted ? 'off' : 'on')))
  if (muted) stopTrainSound()
  else if (trainRunning) startTrainSound()
}

const startTrainSound = () => {
  if (muted || trainOscillator) return
  if (!ensureAudio()) return
  trainOscillator = audioContext.createOscillator()
  trainGain = audioContext.createGain()
  const filter = audioContext.createBiquadFilter()
  trainOscillator.type = 'triangle'
  trainOscillator.frequency.value = 72
  filter.type = 'lowpass'
  filter.frequency.value = 220
  trainGain.gain.value = 0.018
  trainOscillator.connect(filter).connect(trainGain).connect(audioMaster)
  trainOscillator.start()
}
const stopTrainSound = () => {
  if (!trainOscillator) return
  trainOscillator.stop()
  trainOscillator.disconnect()
  trainOscillator = null
  trainGain = null
}

const setTrainRunning = running => {
  trainRunning = running && route.count >= 2
  const button = document.querySelector('#trainToggle')
  button.classList.toggle('running', trainRunning)
  button.querySelector('b').textContent = trainRunning ? 'Stop train' : 'Start train'
  if (trainRunning) startTrainSound()
  else stopTrainSound()
}

const setMode = nextMode => {
  mode = nextMode
  viewport.classList.toggle('simulate-mode', mode === 'simulate')
  document.querySelectorAll('.mode-button').forEach(button => {
    const active = button.dataset.mode === mode
    button.classList.toggle('active', active)
    button.setAttribute('aria-selected', String(active))
  })
  document.querySelector('#buildPanel').hidden = mode !== 'build'
  document.querySelector('#simulatePanel').hidden = mode !== 'simulate'
  gridGroup.visible = mode === 'build'
  highlighter.visible = false
  selectionMarker.visible = mode === 'build' && Boolean(selectedTrackKey)
  if (mode === 'simulate') prepareSimulation()
  else {
    setTrainRunning(false)
    trainParts.forEach(part => { part.visible = false })
  }
}

document.querySelectorAll('.mode-button').forEach(button => button.addEventListener('click', event => {
  event.preventDefault()
  setMode(button.dataset.mode)
}))

const selectPiece = button => {
  selectedPiece = button.dataset.piece
  clearSelectedTrack()
  setEraseMode(false)
  document.querySelectorAll('.piece-card').forEach(card => card.classList.toggle('active', card === button))
  updateRotationUI()
}

const updateRotationUI = () => {
  const active = document.querySelector('.piece-card.active')
  if (!active) return
  active.style.setProperty('--piece-turn', `${selectedRotation * 90}deg`)
  active.dataset.angle = `${selectedRotation * 90}°`
}

const targetedTrackKey = () => {
  if (hoverCell) {
    const key = `${hoverCell.x},${hoverCell.z}`
    if (tracks.has(key) || sceneryItems.has(key)) return key
  }
  return selectedTrackKey && (tracks.has(selectedTrackKey) || sceneryItems.has(selectedTrackKey)) ? selectedTrackKey : null
}

const rotateCurrent = () => {
  const key = targetedTrackKey()
  if (key) {
    const piece = tracks.get(key) || sceneryItems.get(key)
    saveHistory()
    const isTrack = tracks.has(key)
    if (isTrack) trackGroup.remove(piece.mesh)
    else builderScenery.remove(piece.mesh)
    piece.rotation = (piece.rotation + 1) % 4
    if (isTrack) updateTrackElevations()
    else createSceneryMesh(piece)
    selectedRotation = piece.rotation
    selectTrack(key)
    updateRotationUI()
    showToast(`Selected track rotated ${piece.rotation * 90}°`)
    return
  }
  selectedRotation = (selectedRotation + 1) % 4
  updateRotationUI()
  showToast(`Next piece rotated ${selectedRotation * 90}°`)
}

const eraseCurrent = () => {
  const key = targetedTrackKey()
  if (key) {
    removePlacedItem(key)
    setEraseMode(false)
    showToast('Selected track erased')
    return
  }
  setEraseMode(!eraseMode)
  showToast(eraseMode ? 'Erase tool selected. Click a track.' : 'Erase tool off')
}

let paletteDrag = null
let suppressPieceClick = false

document.querySelectorAll('.piece-card').forEach(button => {
  button.addEventListener('click', () => {
    if (suppressPieceClick) return
    selectPiece(button)
  })
  button.addEventListener('pointerdown', event => {
    if (mode !== 'build' || event.button > 0) return
    paletteDrag = { button, startX: event.clientX, startY: event.clientY, active: false, preview: null }
  })
})

window.addEventListener('pointermove', event => {
  if (!paletteDrag) return
  const distance = Math.hypot(event.clientX - paletteDrag.startX, event.clientY - paletteDrag.startY)
  if (!paletteDrag.active && distance > 6) {
    paletteDrag.active = true
    selectPiece(paletteDrag.button)
    paletteDrag.button.classList.add('drag-source')
    paletteDrag.preview = paletteDrag.button.cloneNode(true)
    paletteDrag.preview.className = 'piece-card piece-drag-preview'
    paletteDrag.preview.setAttribute('aria-hidden', 'true')
    document.body.append(paletteDrag.preview)
    document.body.classList.add('dragging-piece')
  }
  if (!paletteDrag.active) return
  paletteDrag.preview.style.left = `${event.clientX}px`
  paletteDrag.preview.style.top = `${event.clientY}px`
  updateHover(event)
})

window.addEventListener('pointerup', event => {
  if (!paletteDrag) return
  const dragged = paletteDrag.active
  if (dragged) {
    const target = document.elementFromPoint(event.clientX, event.clientY)
    if (target === canvas) placeAtHover()
    else showToast('Drop the piece on the play table')
    paletteDrag.button.classList.remove('drag-source')
    paletteDrag.preview.remove()
    document.body.classList.remove('dragging-piece')
    highlighter.visible = false
    hoverCell = null
    suppressPieceClick = true
    setTimeout(() => { suppressPieceClick = false }, 0)
  }
  paletteDrag = null
})
document.querySelector('#rotateButton').addEventListener('click', rotateCurrent)
document.querySelector('#eraseButton').addEventListener('click', eraseCurrent)
document.querySelector('#undoButton').addEventListener('click', undo)
document.querySelector('#trainToggle').addEventListener('click', () => {
  if (route.count < 2) return showToast('Connect at least two track pieces first')
  setTrainRunning(!trainRunning)
})
document.querySelector('#soundButton').addEventListener('click', () => setMuted(!muted))
document.querySelectorAll('[data-sound]').forEach(button => button.addEventListener('click', () => {
  setMuted(button.dataset.sound === 'off')
  showToast(button.dataset.sound === 'off' ? 'All sounds off' : 'Soundscape on')
}))

document.addEventListener('keydown', event => {
  if (mode !== 'build') return
  if (/^[0-9=-]$/.test(event.key)) {
    const card = document.querySelector(`[data-shortcut="${event.key}"]`)
    if (card) {
      selectPiece(card)
      showToast(`${card.querySelector('b').textContent} selected`)
    }
  }
  if (event.key.toLowerCase() === 'r') rotateCurrent()
  if (event.key.toLowerCase() === 'e') eraseCurrent()
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') undo()
})
updateRotationUI()

const createRain = () => {
  const layer = document.querySelector('#weatherLayer')
  for (let i = 0; i < 90; i += 1) {
    const drop = document.createElement('i')
    drop.className = 'rain-drop'
    drop.style.left = `${Math.random() * 110}%`
    drop.style.animationDuration = `${0.45 + Math.random() * 0.45}s`
    drop.style.animationDelay = `${-Math.random() * 2}s`
    drop.style.setProperty('--snow-duration', `${3.2 + Math.random() * 3.6}s`)
    drop.style.setProperty('--snow-drift', `${-90 + Math.random() * 180}px`)
    layer.append(drop)
  }
}
createRain()
document.querySelector('#weatherLayer').style.display = 'none'

const updatePrecipitation = () => {
  const raining = worldState.weather === 'rain'
  const snowing = raining && worldState.ground === 'snow'
  const layer = document.querySelector('#weatherLayer')
  cloudGroup.visible = raining
  layer.style.display = raining ? 'block' : 'none'
  layer.classList.toggle('snowing', snowing)
  renderer.toneMappingExposure = raining ? 0.88 : 1.08
}

const setWorld = (control, value) => {
  worldState[control] = value
  document.querySelectorAll(`[data-world="${control}"]`).forEach(button => button.classList.toggle('active', button.dataset.value === value))
  if (control === 'sky') {
    const night = value === 'night'
    scene.background.set(night ? '#15243b' : '#d9eef1')
    scene.fog.color.set(night ? '#15243b' : '#d9eef1')
    ambient.intensity = night ? 0.55 : 2.2
    ambient.color.set(night ? '#6380aa' : '#fff5d8')
    sun.intensity = night ? 0.22 : 4.1
    nightLamp.intensity = night ? 24 : 0
    starGroup.visible = night
    document.querySelector('.scene-vignette').classList.toggle('night', night)
  }
  if (control === 'weather') {
    updatePrecipitation()
  }
  if (control === 'ground') {
    const snowy = value === 'snow'
    snowGroup.visible = snowy
    groundMaterial.color.set(snowy ? colors.snow : colors.ground)
    updatePrecipitation()
  }
}

document.querySelectorAll('[data-world]').forEach(button => button.addEventListener('click', () => {
  ensureAudio()
  setWorld(button.dataset.world, button.dataset.value)
}))

let toastTimer = null
function showToast(message) {
  const toast = document.querySelector('#toast')
  toast.textContent = message
  toast.classList.add('show')
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => toast.classList.remove('show'), 1600)
}

function updatePieceCount() {
  const count = tracks.size + sceneryItems.size
  document.querySelector('#pieceCount').textContent = `${count} ${count === 1 ? 'piece' : 'pieces'}`
}

const fallbackPoint = (x, z, width, height) => {
  const scale = Math.min(width / 19, height / 14)
  return {
    x: width * 0.52 + (x - z) * scale,
    y: height * 0.52 + (x + z) * scale * 0.48
  }
}

const fallbackPolygon = (context, points) => {
  context.beginPath()
  context.moveTo(points[0].x, points[0].y)
  for (let i = 1; i < points.length; i += 1) context.lineTo(points[i].x, points[i].y)
  context.closePath()
}

const fallbackTrackPath = (context, piece, width, height) => {
  const values = endpoints(piece)
  const heightScale = Math.min(width / 19, height / 14) * 0.7
  if (piece.type === 't-junction' || piece.type === 'x-crossing') {
    const center = fallbackPoint(piece.x, piece.z, width, height)
    center.y -= (piece.elevation || 0) * heightScale
    context.beginPath()
    for (const value of values) {
      const direction = directions[value]
      const end = fallbackPoint(piece.x + direction.x * 0.5, piece.z + direction.z * 0.5, width, height)
      end.y -= ((piece.elevation || 0) + endpointHeight(piece, value)) * heightScale
      context.moveTo(center.x, center.y)
      context.lineTo(end.x, end.y)
    }
    return
  }
  const startDirection = directions[values[0]]
  const endDirection = directions[values[1]]
  const start = fallbackPoint(piece.x + startDirection.x * 0.5, piece.z + startDirection.z * 0.5, width, height)
  const end = fallbackPoint(piece.x + endDirection.x * 0.5, piece.z + endDirection.z * 0.5, width, height)
  start.y -= ((piece.elevation || 0) + endpointHeight(piece, values[0])) * heightScale
  end.y -= ((piece.elevation || 0) + endpointHeight(piece, values[1])) * heightScale
  context.beginPath()
  context.moveTo(start.x, start.y)
  if (piece.type === 'curve') {
    const control = fallbackPoint(piece.x + (startDirection.x + endDirection.x) * 0.5, piece.z + (startDirection.z + endDirection.z) * 0.5, width, height)
    control.y -= (piece.elevation || 0) * heightScale
    context.quadraticCurveTo(control.x, control.y, end.x, end.y)
  } else context.lineTo(end.x, end.y)
}

const drawFallbackTrack = (context, piece, width, height) => {
  fallbackTrackPath(context, piece, width, height)
  context.strokeStyle = '#e0b77c'
  context.lineWidth = 18
  context.lineCap = 'round'
  context.stroke()
  fallbackTrackPath(context, piece, width, height)
  context.strokeStyle = '#69442b'
  context.lineWidth = 3
  context.setLineDash([5, 8])
  context.stroke()
  context.setLineDash([])
  const center = fallbackPoint(piece.x, piece.z, width, height)
  if (piece.type === 'bridge') {
    context.strokeStyle = '#c84e3d'
    context.lineWidth = 5
    context.strokeRect(center.x - 20, center.y - 18, 40, 25)
    context.beginPath()
    context.moveTo(center.x - 20, center.y + 7)
    context.lineTo(center.x, center.y - 18)
    context.lineTo(center.x + 20, center.y + 7)
    context.stroke()
  }
  if (piece.type === 'station') {
    context.fillStyle = '#efb841'
    context.fillRect(center.x + 8, center.y - 26, 34, 27)
    context.fillStyle = '#c84e3d'
    context.beginPath()
    context.moveTo(center.x + 4, center.y - 25)
    context.lineTo(center.x + 25, center.y - 39)
    context.lineTo(center.x + 46, center.y - 25)
    context.closePath()
    context.fill()
    context.fillStyle = '#3d7658'
    context.fillRect(center.x + 20, center.y - 14, 9, 15)
  }
  if (piece.type === 'overpass') {
    context.strokeStyle = '#75837c'
    context.lineWidth = 12
    context.beginPath()
    context.moveTo(center.x - 28, center.y + 15)
    context.lineTo(center.x + 28, center.y - 15)
    context.stroke()
    context.strokeStyle = '#c84e3d'
    context.lineWidth = 4
    context.strokeRect(center.x - 24, center.y - 22, 48, 34)
  }
  if (piece.type === 'track-over') {
    context.strokeStyle = '#c84e3d'
    context.lineWidth = 5
    context.beginPath()
    context.moveTo(center.x - 24, center.y + 17)
    context.lineTo(center.x + 24, center.y - 17)
    context.stroke()
  }
  if (piece.type === 'track-under') {
    context.strokeStyle = '#75837c'
    context.lineWidth = 8
    context.beginPath()
    context.moveTo(center.x - 24, center.y - 17)
    context.lineTo(center.x + 24, center.y + 17)
    context.stroke()
  }
}

const drawFallbackScenery = (context, item, width, height) => {
  const center = fallbackPoint(item.x, item.z, width, height)
  if (item.type === 'tree') {
    context.fillStyle = '#85552f'
    context.fillRect(center.x - 4, center.y - 7, 8, 24)
    context.fillStyle = '#315b3e'
    context.beginPath()
    context.moveTo(center.x, center.y - 55)
    context.lineTo(center.x - 27, center.y + 4)
    context.lineTo(center.x + 27, center.y + 4)
    context.fill()
  }
  if (item.type === 'lake') {
    context.fillStyle = '#69aeb7'
    context.beginPath()
    context.ellipse(center.x, center.y, 33, 19, -0.2, 0, Math.PI * 2)
    context.fill()
    context.strokeStyle = '#e0b77c'
    context.lineWidth = 7
    context.stroke()
  }
  if (item.type === 'house') {
    context.fillStyle = '#efb841'
    context.fillRect(center.x - 24, center.y - 30, 48, 36)
    context.fillStyle = '#c84e3d'
    context.beginPath()
    context.moveTo(center.x - 31, center.y - 29)
    context.lineTo(center.x, center.y - 52)
    context.lineTo(center.x + 31, center.y - 29)
    context.fill()
    context.fillStyle = '#3d7658'
    context.fillRect(center.x - 6, center.y - 14, 12, 20)
  }
}

const drawFallbackLakeConnections = (context, width, height) => {
  const offsets = [[1, 0], [0, 1], [1, 1], [1, -1]]
  context.strokeStyle = '#69aeb7'
  context.lineWidth = 34
  context.lineCap = 'round'
  for (const item of sceneryItems.values()) {
    if (item.type !== 'lake') continue
    for (const [dx, dz] of offsets) {
      const neighbor = sceneryItems.get(`${item.x + dx},${item.z + dz}`)
      if (!neighbor || neighbor.type !== 'lake') continue
      const start = fallbackPoint(item.x, item.z, width, height)
      const end = fallbackPoint(neighbor.x, neighbor.z, width, height)
      context.beginPath()
      context.moveTo(start.x, start.y)
      context.lineTo(end.x, end.y)
      context.stroke()
    }
  }
}

const drawFallbackTrain = (context, width, height) => {
  if (mode !== 'simulate' || route.count < 2) return
  const trainColors = ['#c84e3d', '#efb841', '#46799a', '#3d7658', '#c84e3d', '#f5f4e8', '#46799a', '#efb841', '#3d7658', '#8e332d']
  for (let i = trainColors.length - 1; i >= 0; i -= 1) {
    const sample = sampleRoute(trainDistance - i * 1.15)
    if (!sample) continue
    const point = fallbackPoint(sample.position.x / cellSize, sample.position.z / cellSize, width, height)
    const ahead = fallbackPoint((sample.position.x + sample.tangent.x) / cellSize, (sample.position.z + sample.tangent.z) / cellSize, width, height)
    const heightScale = Math.min(width / 19, height / 14) * 0.7
    point.y -= (sample.position.y - 0.41) * heightScale
    ahead.y -= (sample.position.y - 0.41 + sample.tangent.y) * heightScale
    const angle = Math.atan2(ahead.y - point.y, ahead.x - point.x)
    context.save()
    context.translate(point.x, point.y)
    context.rotate(angle)
    context.fillStyle = 'rgba(45,35,22,.22)'
    context.fillRect(-16, -6, 34, 18)
    context.fillStyle = trainColors[i]
    context.fillRect(-17, -11, 34, 18)
    context.fillStyle = '#29312e'
    context.beginPath()
    context.arc(-10, 9, 5, 0, Math.PI * 2)
    context.arc(10, 9, 5, 0, Math.PI * 2)
    context.fill()
    if (i === 0) {
      context.fillStyle = '#8e332d'
      context.fillRect(6, -21, 9, 12)
      if (trainRunning) {
        for (let puff = 0; puff < 4; puff += 1) {
          const phase = (trainDistance * 2 + puff * 0.27) % 1
          context.fillStyle = `rgba(247,243,231,${0.62 * (1 - phase)})`
          context.beginPath()
          context.arc(18 + phase * 10, -20 - phase * 28, 4 + phase * 7, 0, Math.PI * 2)
          context.fill()
        }
      }
    }
    context.restore()
  }
}

function drawFallbackScene() {
  if (!fallbackActive || !fallbackContext) return
  const width = fallbackCanvas.clientWidth
  const height = fallbackCanvas.clientHeight
  const context = fallbackContext
  const night = document.querySelector('[data-world="sky"][data-value="night"]').classList.contains('active')
  const snowy = document.querySelector('[data-world="ground"][data-value="snow"]').classList.contains('active')
  const sky = context.createLinearGradient(0, 0, 0, height)
  sky.addColorStop(0, night ? '#14233b' : '#cde8eb')
  sky.addColorStop(1, night ? '#314562' : '#f2dfb1')
  context.fillStyle = sky
  context.fillRect(0, 0, width, height)
  if (night) {
    context.fillStyle = '#f5e7b0'
    for (let i = 0; i < 80; i += 1) context.fillRect((i * 83) % width, 82 + (i * 47) % Math.max(1, height - 160), i % 4 === 0 ? 2 : 1, i % 4 === 0 ? 2 : 1)
  }
  const board = [
    fallbackPoint(-5.8, -4.8, width, height),
    fallbackPoint(5.8, -4.8, width, height),
    fallbackPoint(5.8, 4.8, width, height),
    fallbackPoint(-5.8, 4.8, width, height)
  ]
  context.save()
  context.shadowColor = 'rgba(56,37,14,.3)'
  context.shadowBlur = 32
  context.shadowOffsetY = 18
  fallbackPolygon(context, board)
  context.fillStyle = snowy ? '#edf3ee' : '#e3c78e'
  context.fill()
  context.lineWidth = 15
  context.strokeStyle = '#b47c43'
  context.stroke()
  context.restore()
  context.strokeStyle = snowy ? 'rgba(95,120,115,.16)' : 'rgba(109,78,40,.16)'
  context.lineWidth = 1
  for (let x = -5.5; x <= 5.5; x += 1) {
    const start = fallbackPoint(x, -4.5, width, height)
    const end = fallbackPoint(x, 4.5, width, height)
    context.beginPath()
    context.moveTo(start.x, start.y)
    context.lineTo(end.x, end.y)
    context.stroke()
  }
  for (let z = -4.5; z <= 4.5; z += 1) {
    const start = fallbackPoint(-5.5, z, width, height)
    const end = fallbackPoint(5.5, z, width, height)
    context.beginPath()
    context.moveTo(start.x, start.y)
    context.lineTo(end.x, end.y)
    context.stroke()
  }
  const house = fallbackPoint(3.8, -3.25, width, height)
  context.fillStyle = '#46799a'
  context.fillRect(house.x - 28, house.y - 45, 56, 45)
  context.fillStyle = '#efb841'
  context.beginPath()
  context.moveTo(house.x - 36, house.y - 44)
  context.lineTo(house.x, house.y - 72)
  context.lineTo(house.x + 36, house.y - 44)
  context.fill()
  for (const item of sceneryItems.values()) drawFallbackScenery(context, item, width, height)
  drawFallbackLakeConnections(context, width, height)
  for (const piece of tracks.values()) drawFallbackTrack(context, piece, width, height)
  if (mode === 'build' && hoverCell) {
    const point = fallbackPoint(hoverCell.x, hoverCell.z, width, height)
    context.fillStyle = tracks.has(`${hoverCell.x},${hoverCell.z}`) || sceneryItems.has(`${hoverCell.x},${hoverCell.z}`) ? 'rgba(242,189,67,.46)' : 'rgba(61,118,88,.42)'
    context.beginPath()
    context.arc(point.x, point.y, 24, 0, Math.PI * 2)
    context.fill()
  }
  if (mode === 'build' && selectedTrackKey) {
    const item = tracks.get(selectedTrackKey) || sceneryItems.get(selectedTrackKey)
    if (item) {
      const point = fallbackPoint(item.x, item.z, width, height)
      context.strokeStyle = '#f2bd43'
      context.lineWidth = 5
      context.beginPath()
      context.arc(point.x, point.y, 29, 0, Math.PI * 2)
      context.stroke()
    }
  }
  if (snowy) {
    context.fillStyle = 'rgba(255,255,255,.8)'
    for (let i = 0; i < 180; i += 1) {
      const point = fallbackPoint(-5.4 + (i * 37 % 108) / 10, -4.2 + (i * 61 % 84) / 10, width, height)
      context.beginPath()
      context.arc(point.x, point.y, 1 + i % 3, 0, Math.PI * 2)
      context.fill()
    }
  }
  drawFallbackTrain(context, width, height)
}

const updateSoundscape = () => {
  if (!audioContext || muted) return
  const now = audioContext.currentTime
  const rainLevel = worldState.weather === 'rain' ? 0.085 : 0
  const windLevel = worldState.ground === 'snow' ? 0.045 : worldState.sky === 'night' ? 0.025 : 0.012
  const steamLevel = trainRunning ? 0.018 + Math.max(0, Math.sin(trainDistance * 5)) * 0.026 : 0
  rainGain.gain.setTargetAtTime(rainLevel, now, 0.12)
  windGain.gain.setTargetAtTime(windLevel, now, 0.18)
  steamGain.gain.setTargetAtTime(steamLevel, now, 0.08)
  if (now >= nextNatureSound) {
    if (worldState.sky === 'night') {
      playTone(2800, 0.08, 0.018, 'square', 3100)
      playTone(3300, 0.06, 0.012, 'square', 3000)
      nextNatureSound = now + 0.7 + Math.random() * 1.1
    } else if (worldState.weather === 'sunny' && worldState.ground !== 'snow') {
      playTone(1500 + Math.random() * 500, 0.16, 0.022, 'sine', 2700 + Math.random() * 700)
      playTone(1900 + Math.random() * 400, 0.12, 0.016, 'sine', 3100)
      nextNatureSound = now + 1.8 + Math.random() * 2.4
    } else nextNatureSound = now + 1.2
  }
  if (trainRunning && now >= nextPeopleSound) {
    playTone(220, 0.12, 0.012, 'triangle', 280)
    playTone(330, 0.15, 0.009, 'triangle', 250)
    nextPeopleSound = now + 4 + Math.random() * 4
  }
}

const clock = new THREE.Clock()
const resize = () => {
  const width = viewport.clientWidth
  const height = viewport.clientHeight
  renderer.setSize(width, height, false)
  camera.aspect = width / height
  camera.updateProjectionMatrix()
}
window.addEventListener('resize', resize)
resize()

const animate = () => {
  const delta = Math.min(clock.getDelta(), 0.05)
  controls.update()
  if (trainRunning && route.length > 0) {
    const speed = Number(document.querySelector('#speedRange').value)
    trainDistance += delta * speed * 1.55
    trainParts.forEach((part, index) => positionTrainPart(part, trainDistance - index * 1.15))
    if (trainGain) trainGain.gain.value = 0.013 + Math.max(0, Math.sin(trainDistance * 8)) * 0.014
  }
  updateSteam(delta)
  cloudGroup.position.x += delta * 0.12
  if (cloudGroup.position.x > 5) cloudGroup.position.x = -5
  updateSoundscape()
  renderer.render(scene, camera)
  requestAnimationFrame(animate)
}
updatePieceCount()
animate()
