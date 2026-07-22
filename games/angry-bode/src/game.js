import * as THREE from '../vendor/three.module.js'

const canvas = document.querySelector('#game')
const heroCanvas = document.querySelector('#hero-goat')
const startScreen = document.querySelector('#start-screen')
const startButton = document.querySelector('#start-button')
const musicToggle = document.querySelector('#music-toggle')
const musicState = document.querySelector('#music-state')
const pauseScreen = document.querySelector('#pause-screen')
const resumeButton = document.querySelector('#resume-button')
const scoreNode = document.querySelector('#score')
const rageFill = document.querySelector('#rage-fill')
const comboNode = document.querySelector('#combo')
const comboValue = document.querySelector('#combo-value')
const announcement = document.querySelector('#announcement')
const damageFlash = document.querySelector('#damage-flash')

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x8dd8d0)
scene.fog = new THREE.Fog(0x8dd8d0, 35, 78)

const camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.1, 150)
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' })
renderer.outputColorSpace = THREE.SRGBColorSpace
renderer.shadowMap.enabled = false

const clock = new THREE.Clock()
const keys = new Set()
const buildings = []
const citizens = []
const cars = []
const trees = []
const particles = []
const clouds = []
const scenery = []
const attackables = []
const droppings = []
const GOAT_SCALE = 1.56

const palette = {
  ink: 0x191629,
  cream: 0xfff0c7,
  yellow: 0xffd447,
  orange: 0xff6b2c,
  red: 0xe62f32,
  mint: 0x73e0b2,
  blue: 0x42a5d9,
  road: 0x505066,
  sidewalk: 0xb6aa9e,
  grass: 0x5ebb73,
  window: 0x93e6e0
}

const materials = new Map()

function material(color, emissive = 0) {
  const key = `${color}-${emissive}`
  if (!materials.has(key)) {
    materials.set(key, new THREE.MeshLambertMaterial({ color, emissive, flatShading: true }))
  }
  return materials.get(key)
}

function box(width, height, depth, color, x = 0, y = 0, z = 0) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material(color))
  mesh.position.set(x, y, z)
  return mesh
}

function cylinder(radiusTop, radiusBottom, height, color, segments = 6) {
  return new THREE.Mesh(new THREE.CylinderGeometry(radiusTop, radiusBottom, height, segments), material(color))
}

scene.add(new THREE.HemisphereLight(0xfff4cf, 0x594965, 2.2))
const sun = new THREE.DirectionalLight(0xfff0bd, 2.4)
sun.position.set(-18, 30, 22)
scene.add(sun)

function createGround() {
  const ground = box(130, 0.4, 48, palette.grass, 0, -0.25, 0)
  scene.add(ground)

  const road = box(130, 0.12, 12, palette.road, 0, 0.02, 0)
  scene.add(road)

  const northWalk = box(130, 0.22, 3, palette.sidewalk, 0, 0.08, -7.5)
  const southWalk = box(130, 0.22, 3, palette.sidewalk, 0, 0.08, 7.5)
  scene.add(northWalk, southWalk)

  for (let x = -60; x <= 60; x += 6) {
    scene.add(box(3.2, 0.04, 0.3, palette.yellow, x, 0.1, 0))
  }

  for (let x = -57; x <= 57; x += 10) {
    const cross = box(4.8, 0.04, 0.45, palette.cream, x, 0.11, 4.3)
    scene.add(cross)
  }
}

function createSkyline() {
  const colors = [0x6385a3, 0x7d789b, 0x497f88, 0x896f8b]
  for (let i = 0; i < 26; i += 1) {
    const width = 3 + Math.random() * 5
    const height = 7 + Math.random() * 18
    const mesh = box(width, height, 5, colors[i % colors.length], -62 + i * 5.1, height / 2 - 0.1, -24)
    scenery.push(mesh)
    scene.add(mesh)
  }

  for (let i = 0; i < 12; i += 1) {
    const cloud = new THREE.Group()
    const size = 1.3 + Math.random() * 1.5
    cloud.add(box(size * 2.2, size, size, 0xfff0d2, 0, 0, 0))
    cloud.add(box(size * 1.2, size * 1.4, size, 0xfff0d2, -size * .5, size * .45, 0))
    cloud.add(box(size * 1.4, size * 1.2, size, 0xfff0d2, size * .6, size * .3, 0))
    cloud.position.set(-62 + Math.random() * 124, 16 + Math.random() * 13, -31)
    cloud.userData.speed = .35 + Math.random() * .25
    clouds.push(cloud)
    scene.add(cloud)
  }
}

function createTree(x, z) {
  const group = new THREE.Group()
  group.add(box(.45, 2.4, .45, 0x77503c, 0, 1.2, 0))
  const crown = cylinder(0, 1.25, 2.6, 0x26784b, 6)
  crown.position.y = 3
  group.add(crown)
  group.position.set(x, 0, z)
  const tree = { group, alive: true, velocity: new THREE.Vector3(), spin: 0 }
  trees.push(tree)
  scene.add(group)
}

function createLamp(x, z) {
  const lamp = new THREE.Group()
  lamp.add(box(.18, 3.8, .18, palette.ink, 0, 1.9, 0))
  lamp.add(box(.8, .18, .18, palette.ink, .3, 3.75, 0))
  lamp.add(box(.42, .34, .42, palette.yellow, .62, 3.55, 0))
  lamp.position.set(x, 0, z)
  scene.add(lamp)
}

function createStreetDetails() {
  for (let x = -55; x <= 55; x += 11) {
    createTree(x, 10.5)
    createLamp(x + 4.5, 6.1)
    createLamp(x - 2.5, -6.1)
  }

  const sign = new THREE.Group()
  sign.add(box(.35, 5.8, .35, palette.ink, 0, 2.9, 0))
  sign.add(box(6.2, 2.8, .45, palette.orange, 0, 5.4, 0))
  sign.add(box(5.5, .35, .5, palette.yellow, 0, 5.9, .25))
  sign.position.set(-16, 0, 10.5)
  scene.add(sign)

  const tower = new THREE.Group()
  tower.add(box(3.2, 15, 3.2, 0xd6cec0, 0, 7.5, 0))
  tower.add(box(4.2, .8, 4.2, palette.orange, 0, 15, 0))
  const dish = cylinder(0, 1.1, 1.6, palette.cream, 8)
  dish.rotation.z = Math.PI / 2
  dish.position.set(0, 16.1, 0)
  tower.add(dish)
  tower.position.set(52, 0, -17)
  scene.add(tower)
}

function createHighline() {
  const line = new THREE.Group()
  for (let x = -60; x <= 60; x += 8) {
    const pillar = box(.8, 7, .8, 0x45516d, x, 3.5, -20.5)
    line.add(pillar)
    const brace = box(3, .45, .65, 0x45516d, x, 6.5, -20.5)
    line.add(brace)
  }
  line.add(box(130, .55, 1.1, 0x39435d, 0, 7, -20.5))
  line.add(box(130, .18, .16, palette.ink, 0, 7.45, -20.1))
  line.add(box(130, .18, .16, palette.ink, 0, 7.45, -20.9))
  scene.add(line)

  const train = new THREE.Group()
  for (let i = 0; i < 3; i += 1) {
    const car = new THREE.Group()
    car.add(box(5.6, 2.3, 2, i === 0 ? palette.red : palette.cream, -i * 5.9, 0, 0))
    for (let w = 0; w < 3; w += 1) {
      car.add(box(.95, .75, .06, 0x253854, -i * 5.9 - 1.65 + w * 1.55, .3, 1.03))
    }
    train.add(car)
  }
  train.position.set(-66, 8.7, -20.5)
  train.userData.speed = 6.5
  scene.add(train)
  scenery.push(train)
}

const buildingColors = [0xf2b94b, 0xd9664c, 0x4a9faf, 0xe8dbc1, 0x8991bc, 0xc9859f, 0x69a875]

function createBuilding(x, z, width, depth, floors, color, label) {
  const group = new THREE.Group()
  const floorHeight = 2.2
  const height = floors * floorHeight
  const floorGroups = []

  for (let floor = 0; floor < floors; floor += 1) {
    const level = new THREE.Group()
    const y = floor * floorHeight + floorHeight / 2
    const slab = box(width, floorHeight - .08, depth, color, 0, y, 0)
    level.add(slab)

    const windowCount = Math.max(1, Math.floor(width / 1.6))
    for (let i = 0; i < windowCount; i += 1) {
      const windowX = -width / 2 + .8 + i * ((width - 1.6) / Math.max(1, windowCount - 1))
      level.add(box(.66, .78, .08, palette.window, windowX, y + .12, depth / 2 + .045))
      level.add(box(.66, .78, .08, 0x2e5364, windowX, y + .12, -depth / 2 - .045))
    }

    level.add(box(width + .15, .16, depth + .15, palette.ink, 0, floor * floorHeight + .08, 0))
    group.add(level)
    floorGroups.push(level)
  }

  const roof = box(width + .4, .35, depth + .4, palette.ink, 0, height + .16, 0)
  group.add(roof)

  const roofTank = new THREE.Group()
  const tank = cylinder(.7, .7, 1.3, 0xc66a4a, 8)
  tank.position.y = height + 1.3
  roofTank.add(tank)
  roofTank.add(box(.12, 1, .12, palette.ink, -.45, height + .45, 0))
  roofTank.add(box(.12, 1, .12, palette.ink, .45, height + .45, 0))
  if (floors > 3) group.add(roofTank)

  if (label) {
    const sign = new THREE.Group()
    sign.add(box(Math.min(width * .8, 4.6), 1.25, .28, palette.ink, 0, height - 1.15, depth / 2 + .22))
    const stripeColor = label % 2 === 0 ? palette.yellow : palette.mint
    sign.add(box(Math.min(width * .58, 3.5), .25, .3, stripeColor, 0, height - 1.15, depth / 2 + .38))
    group.add(sign)
  }

  group.position.set(x, 0, z)
  const building = {
    group,
    x,
    z,
    width,
    depth,
    floors,
    height,
    health: floors * 42,
    maxHealth: floors * 42,
    floorGroups,
    collapsed: false,
    shake: 0,
    collapseTime: 0,
    damageStage: 0
  }
  group.userData.building = building
  buildings.push(building)
  attackables.push(building)
  scene.add(group)
  return building
}

function populateCity() {
  const rear = [
    [-53, 5.5, 6, 6], [-43, 7, 6, 4], [-31, 5.2, 6, 7], [-20, 7.2, 6, 5],
    [-7, 6.4, 6, 8], [6, 7.2, 6, 4], [18, 6.2, 6, 7], [31, 7, 6, 5], [45, 7.2, 6, 8], [57, 5, 6, 4]
  ]
  const front = [
    [-49, 6.5, 5, 3], [-36, 7, 5, 5], [-23, 6, 5, 4], [-11, 7.2, 5, 3],
    [3, 6.5, 5, 5], [16, 6.4, 5, 3], [28, 7.4, 5, 4], [42, 6.4, 5, 5], [55, 6, 5, 3]
  ]
  rear.forEach((b, i) => createBuilding(b[0], -14, b[1], b[2], b[3], buildingColors[i % buildingColors.length], i))
  front.forEach((b, i) => createBuilding(b[0], -6.8, b[1], b[2], b[3], buildingColors[(i + 3) % buildingColors.length], i + 10))
}

function createGoat() {
  const goat = new THREE.Group()
  const body = box(2.8, 1.45, 1.2, 0xf9f4e6, 0, 1.8, 0)
  goat.add(body)

  const neck = box(.75, 1.45, .8, 0xe8e3d6, 1.1, 2.45, 0)
  neck.rotation.z = -.25
  goat.add(neck)

  const head = new THREE.Group()
  head.add(box(1.25, 1.18, .95, 0xfffbec, 1.65, 3.1, 0))
  head.add(box(.62, .62, .84, 0xded8ca, 2.35, 2.88, 0))
  head.add(box(.2, .16, .18, palette.ink, 2.68, 3, -.23))
  head.add(box(.2, .16, .18, palette.ink, 2.68, 3, .23))

  const eyeMaterial = new THREE.MeshLambertMaterial({ color: 0xff2d2d, emissive: 0xff1d1d, emissiveIntensity: 2 })
  const eyeA = new THREE.Mesh(new THREE.BoxGeometry(.18, .2, .22), eyeMaterial)
  const eyeB = eyeA.clone()
  eyeA.position.set(2.27, 3.32, -.48)
  eyeB.position.set(2.27, 3.32, .48)
  head.add(eyeA, eyeB)

  const earA = box(.58, .18, .36, 0xe4b8ad, 1.55, 3.55, -.63)
  const earB = earA.clone()
  earB.position.z = .63
  head.add(earA, earB)

  const hornA = cylinder(.08, .27, 1.05, 0xd7b66f, 5)
  const hornB = hornA.clone()
  hornA.position.set(1.3, 4.03, -.32)
  hornB.position.set(1.3, 4.03, .32)
  hornA.rotation.z = -.32
  hornB.rotation.z = -.32
  head.add(hornA, hornB)

  const beard = cylinder(0, .42, 1.05, 0xf4eee0, 5)
  beard.position.set(1.98, 2.28, 0)
  head.add(beard)
  goat.add(head)

  const legs = []
  const legPositions = [[-.95, -.38], [-.95, .38], [.88, -.38], [.88, .38]]
  legPositions.forEach(([x, z]) => {
    const leg = new THREE.Group()
    leg.add(box(.33, 1.25, .34, 0xf2ede0, 0, -.4, 0))
    leg.add(box(.42, .28, .48, palette.ink, .1, -1.02, 0))
    leg.position.set(x, 1.25, z)
    legs.push(leg)
    goat.add(leg)
  })

  const tail = box(.82, .38, .52, 0xf4eee0, -1.68, 2.25, 0)
  tail.rotation.z = -.55
  goat.add(tail)
  goat.scale.setScalar(GOAT_SCALE)
  goat.position.set(-4, .15, 3)
  scene.add(goat)

  return { group: goat, head, body, legs, tail, eyeA, eyeB }
}

const goatVisual = createGoat()

const heroScene = new THREE.Scene()
const heroCamera = new THREE.PerspectiveCamera(38, 1, .1, 80)
const heroRenderer = new THREE.WebGLRenderer({ canvas: heroCanvas, antialias: false, alpha: true })
heroRenderer.outputColorSpace = THREE.SRGBColorSpace
heroRenderer.setClearColor(0x000000, 0)
heroScene.add(new THREE.HemisphereLight(0xfff4cf, 0x6a3352, 3.2))
const heroLight = new THREE.DirectionalLight(0xffd16f, 3)
heroLight.position.set(-5, 12, 9)
heroScene.add(heroLight)
const heroGoat = goatVisual.group.clone(true)
heroGoat.position.set(-1.5, -1.2, 0)
heroGoat.rotation.y = -.18
heroScene.add(heroGoat)
const heroLaserMaterial = new THREE.MeshBasicMaterial({ color: 0xff2020, transparent: true, opacity: .92 })
const heroLaserCoreMaterial = new THREE.MeshBasicMaterial({ color: 0xffffdb })
const heroLasers = new THREE.Group()
;[-.75, .75].forEach(z => {
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(.13, .13, 15, 5), heroLaserMaterial)
  const core = new THREE.Mesh(new THREE.CylinderGeometry(.05, .05, 15, 5), heroLaserCoreMaterial)
  beam.position.set(11, 4, z)
  core.position.copy(beam.position)
  beam.rotation.z = -Math.PI / 2
  core.rotation.z = -Math.PI / 2
  heroLasers.add(beam, core)
})
heroScene.add(heroLasers)
heroCamera.position.set(12, 9, 16)
heroCamera.lookAt(0, 3.2, 0)

const player = {
  position: goatVisual.group.position.clone(),
  velocity: new THREE.Vector3(),
  facing: 1,
  grounded: true,
  climbing: false,
  kickTime: 0,
  kickCooldown: 0,
  burpTime: 0,
  burpCooldown: 0,
  laser: false,
  rage: 100,
  score: 0,
  combo: 1,
  comboTimer: 0,
  walkTime: 0,
  poopTimer: 6 + Math.random() * 5,
  poopTime: 0
}

const beamMaterial = new THREE.MeshBasicMaterial({ color: 0xff2020, transparent: true, opacity: .88 })
const beamCoreMaterial = new THREE.MeshBasicMaterial({ color: 0xffffa1 })
const laserGroup = new THREE.Group()
const beamA = new THREE.Mesh(new THREE.CylinderGeometry(.09, .09, 1, 5), beamMaterial)
const beamB = beamA.clone()
const coreA = new THREE.Mesh(new THREE.CylinderGeometry(.035, .035, 1, 5), beamCoreMaterial)
const coreB = coreA.clone()
laserGroup.add(beamA, beamB, coreA, coreB)
laserGroup.visible = false
scene.add(laserGroup)

function createCitizen(x, z, shirtColor) {
  const group = new THREE.Group()
  const skinColors = [0x6f4638, 0xa96f4d, 0xd6a179, 0xf0c8a4]
  const skin = skinColors[Math.floor(Math.random() * skinColors.length)]
  const head = new THREE.Group()
  head.add(box(.42, .45, .42, skin, 0, 1.85, 0))
  head.add(box(.48, .16, .46, [0x28202c, 0x593824, 0x9b6b3c][Math.floor(Math.random() * 3)], 0, 2.07, 0))
  group.add(head)
  group.add(box(.58, .8, .36, shirtColor, 0, 1.2, 0))
  const legA = box(.2, .72, .22, 0x283d65, -.16, .45, 0)
  const legB = box(.2, .72, .22, 0x283d65, .16, .45, 0)
  group.add(legA, legB)
  group.scale.setScalar(.72)
  group.position.set(x, .12, z)
  group.rotation.y = Math.random() > .5 ? 0 : Math.PI
  scene.add(group)
  citizens.push({
    group,
    legA,
    legB,
    head,
    speed: 2.1 + Math.random() * 1.6,
    phase: Math.random() * Math.PI * 2,
    direction: Math.random() > .5 ? 1 : -1,
    stunned: 0,
    alive: true
  })
}

function populateCitizens() {
  const shirts = [palette.orange, palette.blue, palette.yellow, palette.mint, 0xd25886]
  for (let i = 0; i < 28; i += 1) {
    const z = Math.random() > .5 ? -5.1 : 4.5 + Math.random() * 1.2
    createCitizen(-56 + Math.random() * 112, z, shirts[i % shirts.length])
  }
}

function createCar(x, z, color) {
  const group = new THREE.Group()
  group.add(box(3.4, .8, 1.5, color, 0, .65, 0))
  group.add(box(1.8, .7, 1.3, color, -.2, 1.32, 0))
  group.add(box(.65, .48, 1.36, 0x9ad8dc, -.55, 1.38, 0))
  const wheelMaterial = material(palette.ink)
  ;[-1.05, 1.05].forEach(wx => {
    ;[-.77, .77].forEach(wz => {
      const wheel = new THREE.Mesh(new THREE.CylinderGeometry(.34, .34, .18, 8), wheelMaterial)
      wheel.rotation.x = Math.PI / 2
      wheel.position.set(wx, .35, wz)
      group.add(wheel)
    })
  })
  group.position.set(x, 0, z)
  const car = {
    group,
    color,
    speed: 1.7 + Math.random() * 1.4,
    direction: Math.random() > .5 ? 1 : -1,
    wrecked: false,
    wreckTime: 0,
    velocity: new THREE.Vector3(),
    spin: new THREE.Vector3()
  }
  group.userData.car = car
  scene.add(group)
  cars.push(car)
}

function populateCars() {
  createCar(-42, -2.2, palette.red)
  createCar(-17, 2.1, palette.yellow)
  createCar(14, -2.2, palette.blue)
  createCar(39, 2.1, palette.mint)
}

function spawnParticle(position, color, velocity, scale = .25, life = 1) {
  const mesh = box(scale, scale, scale, color)
  mesh.position.copy(position)
  mesh.rotation.set(Math.random() * 3, Math.random() * 3, Math.random() * 3)
  scene.add(mesh)
  particles.push({ mesh, velocity: velocity.clone(), life, maxLife: life, gravity: 9 })
}

function spawnBurst(position, color, count = 10, force = 5) {
  for (let i = 0; i < count; i += 1) {
    const velocity = new THREE.Vector3((Math.random() - .5) * force, Math.random() * force + 1.5, (Math.random() - .5) * force)
    spawnParticle(position, color, velocity, .16 + Math.random() * .32, .65 + Math.random() * .7)
  }
}

function createBurpWave() {
  const geometry = new THREE.TorusGeometry(1, .12, 4, 12)
  const waveMaterial = new THREE.MeshBasicMaterial({ color: 0x8eff52, transparent: true, opacity: .8 })
  const wave = new THREE.Mesh(geometry, waveMaterial)
  wave.rotation.y = Math.PI / 2
  wave.position.set(player.position.x + player.facing * 2, player.position.y + 2.3, player.position.z)
  wave.scale.x = player.facing
  scene.add(wave)
  particles.push({ mesh: wave, velocity: new THREE.Vector3(player.facing * 6, .3, 0), life: .7, maxLife: .7, gravity: 0, grow: true })
}

function dropPoop() {
  const pile = new THREE.Group()
  const poopColor = 0x4a2d22
  pile.add(box(.8, .55, .75, poopColor, 0, .3, 0))
  pile.add(box(.58, .48, .56, 0x62402b, -.18, .72, .05))
  pile.add(box(.34, .32, .36, poopColor, .06, 1.02, 0))
  pile.rotation.y = Math.random() * Math.PI
  pile.position.set(player.position.x - player.facing * 2.7, player.position.y, player.position.z)
  scene.add(pile)
  droppings.push(pile)
  if (droppings.length > 14) scene.remove(droppings.shift())
  player.poopTime = .7
  announce('PLOP!')
  tone(82, .16, 'square', .045, -35)
}

function makeDebris(building, count, power) {
  const y = Math.max(1, Math.min(building.height, player.position.y + 1.5))
  const colors = [building.group.children[0].children[0].material.color.getHex(), palette.ink, palette.window]
  for (let i = 0; i < count; i += 1) {
    const position = new THREE.Vector3(
      building.x + (Math.random() - .5) * building.width,
      y + (Math.random() - .5) * 2,
      building.z + (Math.random() - .5) * building.depth
    )
    const velocity = new THREE.Vector3((Math.random() - .5) * power + player.facing * 2, Math.random() * power, (Math.random() - .5) * power)
    spawnParticle(position, colors[i % colors.length], velocity, .25 + Math.random() * .48, 1 + Math.random() * 1.4)
  }
}

function formatScore(value) {
  return Math.floor(value).toString().padStart(6, '0')
}

function updateHud() {
  scoreNode.textContent = formatScore(player.score)
  rageFill.style.width = `${Math.max(0, player.rage)}%`
  comboValue.textContent = `x${player.combo}`
  comboNode.classList.toggle('active', player.combo > 1)
}

let announcementTimer = 0

function announce(text) {
  announcement.textContent = text
  announcement.classList.remove('show')
  void announcement.offsetWidth
  announcement.classList.add('show')
  announcementTimer = .9
}

function flash() {
  damageFlash.classList.remove('hit')
  void damageFlash.offsetWidth
  damageFlash.classList.add('hit')
}

let audioContext
let banjoGain
let musicEnabled = true
let banjoStep = 0
let banjoNextTime = 0
const banjoRoll = [4, 2, 0, 1, 2, 4, 1, 2, 4, 2, 0, 1, 3, 4, 1, 2]
const banjoChords = [
  [392, 146.83, 196, 246.94, 293.66],
  [392, 130.81, 196, 261.63, 329.63],
  [440, 146.83, 220, 293.66, 369.99],
  [392, 164.81, 196, 246.94, 329.63]
]

function initAudio() {
  if (!audioContext) {
    audioContext = new AudioContext()
    banjoGain = audioContext.createGain()
    banjoGain.gain.value = 0
    banjoGain.connect(audioContext.destination)
  }
  if (audioContext.state === 'suspended') audioContext.resume()
}

function setBanjoVolume() {
  if (!audioContext || !banjoGain) return
  const volume = musicEnabled && started && !paused ? .26 : .0001
  banjoGain.gain.cancelScheduledValues(audioContext.currentTime)
  banjoGain.gain.setTargetAtTime(volume, audioContext.currentTime, .035)
  if (volume > .001) banjoNextTime = audioContext.currentTime
}

function pluckBanjo(frequency, time, accent = 1) {
  if (!audioContext || !banjoGain) return
  const sampleRate = audioContext.sampleRate
  const duration = .62
  const length = Math.floor(sampleRate * duration)
  const delay = Math.max(2, Math.round(sampleRate / frequency))
  const buffer = audioContext.createBuffer(1, length, sampleRate)
  const data = buffer.getChannelData(0)
  for (let i = 0; i < delay; i += 1) data[i] = (Math.random() * 2 - 1) * (.85 - i / delay * .18)
  for (let i = delay; i < length; i += 1) {
    const next = i - delay + 1
    data[i] = (data[i - delay] + data[next]) * .4975
    data[i] *= Math.exp(-i / (sampleRate * 1.8))
  }
  const source = audioContext.createBufferSource()
  const highpass = audioContext.createBiquadFilter()
  const body = audioContext.createBiquadFilter()
  const lowpass = audioContext.createBiquadFilter()
  const gain = audioContext.createGain()
  source.buffer = buffer
  highpass.type = 'highpass'
  highpass.frequency.value = 110
  body.type = 'peaking'
  body.frequency.value = 920
  body.Q.value = 1.3
  body.gain.value = 5
  lowpass.type = 'lowpass'
  lowpass.frequency.value = 5200
  lowpass.Q.value = .4
  gain.gain.setValueAtTime(.18 * accent, time)
  gain.gain.exponentialRampToValueAtTime(.0001, time + duration)
  source.connect(highpass).connect(body).connect(lowpass).connect(gain).connect(banjoGain)
  source.start(time)
  source.stop(time + duration)
}

function updateBanjo() {
  if (!audioContext || !musicEnabled || paused) return
  const beat = 60 / 168 / 4
  while (banjoNextTime < audioContext.currentTime + .12) {
    const section = Math.floor(banjoStep / 32) % banjoChords.length
    const chord = banjoChords[section]
    const string = banjoRoll[banjoStep % banjoRoll.length]
    pluckBanjo(chord[string], banjoNextTime, banjoStep % 4 === 0 ? 1.15 : .78)
    if (banjoStep % 16 === 0) {
      chord.slice(1).forEach((frequency, index) => pluckBanjo(frequency, banjoNextTime + index * .006, .42))
    }
    banjoStep += 1
    banjoNextTime += beat
  }
}

function tone(frequency, duration, type = 'square', volume = .05, slide = 0) {
  if (!audioContext) return
  const oscillator = audioContext.createOscillator()
  const gain = audioContext.createGain()
  oscillator.type = type
  oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime)
  oscillator.frequency.linearRampToValueAtTime(Math.max(25, frequency + slide), audioContext.currentTime + duration)
  gain.gain.setValueAtTime(volume, audioContext.currentTime)
  gain.gain.exponentialRampToValueAtTime(.001, audioContext.currentTime + duration)
  oscillator.connect(gain).connect(audioContext.destination)
  oscillator.start()
  oscillator.stop(audioContext.currentTime + duration)
}

function noise(duration, volume = .06) {
  if (!audioContext) return
  const length = Math.floor(audioContext.sampleRate * duration)
  const buffer = audioContext.createBuffer(1, length, audioContext.sampleRate)
  const data = buffer.getChannelData(0)
  for (let i = 0; i < length; i += 1) data[i] = Math.random() * 2 - 1
  const source = audioContext.createBufferSource()
  const gain = audioContext.createGain()
  source.buffer = buffer
  gain.gain.setValueAtTime(volume, audioContext.currentTime)
  gain.gain.exponentialRampToValueAtTime(.001, audioContext.currentTime + duration)
  source.connect(gain).connect(audioContext.destination)
  source.start()
}

function addScore(amount) {
  player.comboTimer = 2.4
  player.score += amount * player.combo
  player.rage = Math.min(100, player.rage + amount * .025)
  updateHud()
}

function damageBuilding(building, amount, attack) {
  if (!building || building.collapsed) return false
  building.health -= amount
  building.shake = Math.min(.35, building.shake + amount * .006)
  addScore(amount * 8)
  const nextStage = Math.floor((1 - building.health / building.maxHealth) * building.floors)

  if (nextStage > building.damageStage) {
    building.damageStage = nextStage
    const floorIndex = building.floors - nextStage
    if (building.floorGroups[floorIndex]) {
      building.floorGroups[floorIndex].rotation.z = (Math.random() - .5) * .08
    }
    makeDebris(building, 7, 4.5)
    flash()
    tone(95, .1, 'square', .05, -25)
  }

  if (building.health <= 0) {
    building.collapsed = true
    building.collapseTime = 0
    player.combo = Math.min(9, player.combo + 1)
    player.comboTimer = 4
    addScore(building.floors * 650)
    makeDebris(building, 30, 8)
    announce(attack === 'laser' ? 'MELTDOWN!' : 'BAAAAM!')
    noise(.45, .12)
    tone(62, .5, 'sawtooth', .08, -32)
  }
  return true
}

function nearestBuilding(maxDistance) {
  let best = null
  let bestDistance = maxDistance
  for (const building of buildings) {
    if (building.collapsed) continue
    if (player.position.y > building.height + 1.5) continue
    const dx = (building.x - player.position.x) * player.facing
    const zDistance = Math.abs(building.z - player.position.z) - building.depth / 2
    const faceDistance = dx - building.width / 2
    if (dx > 0 && faceDistance < bestDistance && zDistance < 1.4) {
      bestDistance = Math.max(.4, faceDistance)
      best = building
    }
  }
  return { building: best, distance: bestDistance }
}

let screamCooldown = 0

function scream() {
  if (screamCooldown > 0) return
  screamCooldown = .22
  tone(760 + Math.random() * 160, .36, 'sawtooth', .06, -410)
  tone(1040 + Math.random() * 180, .24, 'square', .025, -520)
  announce('AAAAAA!')
}

function knockCitizen(citizen, force = 5) {
  if (!citizen.alive) return
  citizen.alive = false
  const origin = citizen.group.position.clone().add(new THREE.Vector3(0, 1, 0))
  spawnBurst(origin, palette.yellow, 9, force)
  citizen.velocity = new THREE.Vector3(player.facing * force, force * .85, (Math.random() - .5) * force)
  citizen.life = 2.6
  addScore(125)
  scream()
}

function hitCar(car, force = 8) {
  if (!car || car.wrecked) return false
  car.wrecked = true
  car.wreckTime = 0
  car.velocity.set(player.facing * force, force * .72, (Math.random() - .5) * force * .35)
  car.spin.set((Math.random() - .5) * 4, (Math.random() - .5) * 5, -player.facing * (3 + Math.random() * 3))
  const origin = car.group.position.clone().add(new THREE.Vector3(0, 1, 0))
  spawnBurst(origin, palette.orange, 18, 8)
  spawnBurst(origin, palette.ink, 10, 6)
  addScore(900)
  announce('KABOOM!')
  flash()
  noise(.32, .12)
  tone(58, .45, 'sawtooth', .1, -28)
  return true
}

function destroyTree(tree, force = 7) {
  if (!tree || !tree.alive) return false
  tree.alive = false
  tree.velocity.set(player.facing * force, force * .55, (Math.random() - .5) * force)
  tree.spin = player.facing * (2.5 + Math.random() * 2)
  const origin = tree.group.position.clone().add(new THREE.Vector3(0, 2, 0))
  spawnBurst(origin, 0x26784b, 14, 6)
  spawnBurst(origin, 0x77503c, 7, 5)
  addScore(350)
  announce('CRUNCH!')
  noise(.18, .08)
  tone(74, .28, 'square', .07, -30)
  return true
}

function trampleNearby(radius, force) {
  if (player.position.y > 1.4) return
  let crushed = false
  for (const citizen of citizens) {
    if (!citizen.alive) continue
    const dx = citizen.group.position.x - player.position.x
    const dz = citizen.group.position.z - player.position.z
    if (Math.hypot(dx, dz) < radius) {
      knockCitizen(citizen, force)
      crushed = true
    }
  }
  for (const car of cars) {
    if (car.wrecked) continue
    const dx = car.group.position.x - player.position.x
    const dz = car.group.position.z - player.position.z
    if (Math.hypot(dx, dz) < radius + 1.2) {
      hitCar(car, force)
      crushed = true
    }
  }
  for (const tree of trees) {
    if (!tree.alive) continue
    const dx = tree.group.position.x - player.position.x
    const dz = tree.group.position.z - player.position.z
    if (Math.hypot(dx, dz) < radius + .5) {
      destroyTree(tree, force)
      crushed = true
    }
  }
  if (crushed && force > 10) {
    announce('STOMP!')
    flash()
    noise(.22, .1)
  }
}

function carsInAttack(distance, spread) {
  for (const car of cars) {
    if (car.wrecked) continue
    const delta = car.group.position.clone().sub(player.position)
    if (delta.x * player.facing > 0 && Math.abs(delta.z) < spread && delta.length() < distance) hitCar(car, 9)
  }
}

function citizensInAttack(distance, spread) {
  for (const citizen of citizens) {
    if (!citizen.alive) continue
    const delta = citizen.group.position.clone().sub(player.position)
    if (delta.x * player.facing > 0 && Math.abs(delta.z) < spread && delta.length() < distance) knockCitizen(citizen)
  }
}

function doKick() {
  if (player.kickCooldown > 0 || player.burpTime > 0) return
  player.kickTime = .38
  player.kickCooldown = .55
  const target = nearestBuilding(6.5)
  if (target.building) {
    damageBuilding(target.building, 38, 'kick')
    announce('KRAK!')
    const hitPoint = new THREE.Vector3(player.position.x + player.facing * 4.2, player.position.y + 2.8, player.position.z)
    spawnBurst(hitPoint, palette.yellow, 12, 6)
    noise(.12, .09)
    tone(86, .14, 'square', .08, -20)
  } else {
    tone(180, .08, 'square', .035, -40)
  }
  citizensInAttack(7, 2.8)
  carsInAttack(8, 3)
}

function doBurp() {
  if (player.burpCooldown > 0 || player.kickTime > 0) return
  player.burpTime = .6
  player.burpCooldown = 2
  player.rage = Math.max(0, player.rage - 6)
  createBurpWave()
  const target = nearestBuilding(8)
  if (target.building) damageBuilding(target.building, 17, 'burp')
  citizensInAttack(8, 3.2)
  carsInAttack(9, 3.8)
  announce('BUUURP!')
  tone(105, .5, 'sawtooth', .1, -65)
  tone(64, .55, 'square', .045, 18)
}

function nearestLaserTarget(maxDistance) {
  const buildingHit = nearestBuilding(maxDistance)
  let hit = buildingHit.building ? {
    type: 'building',
    target: buildingHit.building,
    distance: buildingHit.distance,
    point: new THREE.Vector3(
      player.position.x + player.facing * buildingHit.distance,
      Math.min(buildingHit.building.height - .4, player.position.y + 5.2),
      THREE.MathUtils.clamp(player.position.z, buildingHit.building.z - buildingHit.building.depth / 2, buildingHit.building.z + buildingHit.building.depth / 2)
    )
  } : null

  for (const car of cars) {
    if (car.wrecked) continue
    const dx = (car.group.position.x - player.position.x) * player.facing
    if (dx > 0 && dx < (hit?.distance ?? maxDistance) && Math.abs(car.group.position.z - player.position.z) < 2.1) {
      hit = { type: 'car', target: car, distance: dx, point: car.group.position.clone().add(new THREE.Vector3(0, 1, 0)) }
    }
  }

  for (const citizen of citizens) {
    if (!citizen.alive) continue
    const dx = (citizen.group.position.x - player.position.x) * player.facing
    if (dx > 0 && dx < (hit?.distance ?? maxDistance) && Math.abs(citizen.group.position.z - player.position.z) < 1.8) {
      hit = { type: 'citizen', target: citizen, distance: dx, point: citizen.group.position.clone().add(new THREE.Vector3(0, 1, 0)) }
    }
  }
  return hit
}

function placeBeam(beam, start, end) {
  const direction = end.clone().sub(start)
  const length = direction.length()
  beam.position.copy(start).add(end).multiplyScalar(.5)
  beam.scale.set(1, length, 1)
  beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize())
}

function updateLaser(dt) {
  const active = keys.has('KeyE') && player.rage > 0 && player.kickTime <= 0
  player.laser = active
  laserGroup.visible = active
  goatVisual.eyeA.material.emissiveIntensity = active ? 5 : 2
  goatVisual.eyeB.material.emissiveIntensity = active ? 5 : 2
  if (!active) return

  player.rage = Math.max(0, player.rage - dt * 9)
  const hit = nearestLaserTarget(60)
  const startX = player.position.x + player.facing * 3.55
  const startY = player.position.y + 5.2
  const jitter = Math.sin(performance.now() * .08) * .025
  const end = hit?.point ?? new THREE.Vector3(startX + player.facing * 60, startY, player.position.z)
  const startA = new THREE.Vector3(startX, startY + jitter, player.position.z - .75)
  const startB = new THREE.Vector3(startX, startY + jitter, player.position.z + .75)

  placeBeam(beamA, startA, end)
  placeBeam(beamB, startB, end)
  placeBeam(coreA, startA, end)
  placeBeam(coreB, startB, end)

  if (hit) {
    if (hit.type === 'building') damageBuilding(hit.target, dt * 28, 'laser')
    if (hit.type === 'car') hitCar(hit.target, 11)
    if (hit.type === 'citizen') knockCitizen(hit.target, 8)
    if (Math.random() < dt * 24) spawnBurst(end, palette.red, 2, 3)
  }
  if (Math.random() < dt * 6) tone(230 + Math.random() * 35, .07, 'sawtooth', .018, 30)
}

function touchingBuilding() {
  let closest = null
  let closestDistance = 1.8
  for (const building of buildings) {
    if (building.collapsed || player.position.y > building.height + .7) continue
    const dx = Math.max(Math.abs(player.position.x - building.x) - building.width / 2, 0)
    const dz = Math.max(Math.abs(player.position.z - building.z) - building.depth / 2, 0)
    const distance = Math.hypot(dx, dz)
    if (distance < closestDistance) {
      closestDistance = distance
      closest = building
    }
  }
  return closest
}

function supportHeight(x, z, currentY) {
  let height = .15
  for (const building of buildings) {
    if (building.collapsed) continue
    const insideX = Math.abs(x - building.x) < building.width / 2 - .25
    const insideZ = Math.abs(z - building.z) < building.depth / 2 - .25
    if (insideX && insideZ && currentY >= building.height - .4) height = Math.max(height, building.height + .18)
  }
  return height
}

function resolveBuildingCollision(previous) {
  for (const building of buildings) {
    if (building.collapsed || player.position.y >= building.height - .1) continue
    const overlapX = Math.abs(player.position.x - building.x) < building.width / 2 + 1.1
    const overlapZ = Math.abs(player.position.z - building.z) < building.depth / 2 + .9
    if (overlapX && overlapZ) {
      player.position.x = previous.x
      player.position.z = previous.z
      return building
    }
  }
  return null
}

function updatePlayer(dt) {
  screamCooldown = Math.max(0, screamCooldown - dt)
  player.kickCooldown = Math.max(0, player.kickCooldown - dt)
  player.burpCooldown = Math.max(0, player.burpCooldown - dt)
  player.kickTime = Math.max(0, player.kickTime - dt)
  player.burpTime = Math.max(0, player.burpTime - dt)
  player.poopTime = Math.max(0, player.poopTime - dt)
  player.comboTimer -= dt
  if (player.comboTimer <= 0 && player.combo > 1) {
    player.combo = 1
    updateHud()
  }

  const previous = player.position.clone()
  let stompLanding = false
  const touch = touchingBuilding()
  const wantsClimb = keys.has('ArrowUp') && touch && player.position.y < touch.height - .1
  player.climbing = Boolean(wantsClimb)
  let moveX = 0
  let moveZ = 0

  if (keys.has('KeyA') || keys.has('ArrowLeft')) moveX -= 1
  if (keys.has('KeyD') || keys.has('ArrowRight')) moveX += 1
  if (!wantsClimb && keys.has('ArrowUp')) moveZ -= 1
  if (keys.has('ArrowDown')) moveZ += 1

  if (moveX !== 0) player.facing = Math.sign(moveX)
  const moveLength = Math.hypot(moveX, moveZ) || 1
  const speed = player.grounded ? 5.5 : 4.1
  player.position.x += moveX / moveLength * speed * dt
  player.position.z += moveZ / moveLength * speed * dt
  player.position.x = THREE.MathUtils.clamp(player.position.x, -61, 61)
  player.position.z = THREE.MathUtils.clamp(player.position.z, -17, 17)

  const collision = resolveBuildingCollision(previous)
  if (collision && wantsClimb) {
    player.position.copy(previous)
    player.climbing = true
  }

  if (player.climbing) {
    player.velocity.y = 0
    player.position.y += 4.2 * dt
    player.grounded = false
    if (touch && player.position.y >= touch.height + .18) {
      player.position.y = touch.height + .18
      player.position.x = THREE.MathUtils.clamp(player.position.x, touch.x - touch.width / 2 + .7, touch.x + touch.width / 2 - .7)
      player.position.z = THREE.MathUtils.clamp(player.position.z, touch.z - touch.depth / 2 + .7, touch.z + touch.depth / 2 - .7)
      player.climbing = false
      player.grounded = true
    }
  } else {
    player.velocity.y -= 15 * dt
    player.position.y += player.velocity.y * dt
    const floor = supportHeight(player.position.x, player.position.z, previous.y)
    if (player.position.y <= floor) {
      stompLanding = !player.grounded && player.velocity.y < -3
      if (!player.grounded && player.velocity.y < -6) {
        spawnBurst(new THREE.Vector3(player.position.x, floor, player.position.z), 0xd7c3a3, 8, 2.5)
        tone(72, .1, 'square', .05, -12)
      }
      player.position.y = floor
      player.velocity.y = 0
      player.grounded = true
    } else {
      player.grounded = false
    }
  }

  const moving = Math.abs(moveX) + Math.abs(moveZ) > 0
  if (stompLanding) trampleNearby(4.4, 13)
  else if (moving && player.grounded) trampleNearby(2.7, 7)
  if (moving) player.walkTime += dt * 10
  if (moving && player.grounded) {
    player.poopTimer -= dt
    if (player.poopTimer <= 0) {
      dropPoop()
      player.poopTimer = 8 + Math.random() * 7
    }
  }
  const stride = moving && player.grounded ? Math.sin(player.walkTime) * .72 : 0
  const stomp = moving && player.grounded ? Math.abs(Math.sin(player.walkTime)) * .11 : 0
  goatVisual.legs[0].rotation.z = stride
  goatVisual.legs[1].rotation.z = -stride
  goatVisual.legs[2].rotation.z = -stride
  goatVisual.legs[3].rotation.z = stride
  goatVisual.tail.rotation.z = (player.poopTime > 0 ? .35 : -.55) + Math.sin(performance.now() * .018) * .3
  goatVisual.body.rotation.z = moving ? Math.sin(player.walkTime * 2) * .045 : 0
  goatVisual.body.position.y = 1.8 + stomp
  goatVisual.head.position.y = stomp * .7

  if (player.kickTime > 0) {
    const phase = Math.sin((1 - player.kickTime / .38) * Math.PI)
    goatVisual.legs[0].rotation.z = -phase * 1.7
    goatVisual.legs[1].rotation.z = -phase * 1.7
    goatVisual.body.rotation.z = phase * .14
  }
  goatVisual.head.rotation.z = player.burpTime > 0 ? Math.sin(player.burpTime * 17) * .13 : 0

  goatVisual.group.position.copy(player.position)
  goatVisual.group.scale.x = GOAT_SCALE * player.facing
  updateLaser(dt)
  updateHud()
}

function jump() {
  if (player.grounded || player.climbing) {
    player.velocity.y = 7.8
    player.grounded = false
    player.climbing = false
    tone(190, .15, 'square', .05, 180)
  }
}

function updateBuildings(dt) {
  for (const building of buildings) {
    if (building.shake > 0 && !building.collapsed) {
      building.group.position.x = building.x + (Math.random() - .5) * building.shake
      building.group.position.z = building.z + (Math.random() - .5) * building.shake
      building.shake = Math.max(0, building.shake - dt * .75)
    } else if (!building.collapsed) {
      building.group.position.x = building.x
      building.group.position.z = building.z
    }

    if (building.collapsed) {
      building.collapseTime += dt
      building.group.position.y -= dt * (1.4 + building.collapseTime * 2.2)
      building.group.rotation.z += dt * .08 * Math.sign(building.x - player.position.x || 1)
      building.group.scale.y = Math.max(.05, 1 - building.collapseTime * .5)
      if (building.group.position.y < -building.height) building.group.visible = false
    }
  }
}

function updateCitizens(dt) {
  for (const citizen of citizens) {
    if (!citizen.alive) {
      citizen.life -= dt
      citizen.velocity.y -= 13 * dt
      citizen.group.position.addScaledVector(citizen.velocity, dt)
      citizen.group.rotation.x += dt * 8
      citizen.group.rotation.z += dt * 6
      if (citizen.group.position.y < .1) {
        citizen.group.position.y = .1
        citizen.velocity.y *= -.28
        citizen.velocity.x *= .72
        citizen.velocity.z *= .72
      }
      if (citizen.life <= 0) citizen.group.visible = false
      continue
    }
    const distance = citizen.group.position.distanceTo(player.position)
    if (distance < 13) citizen.direction = citizen.group.position.x > player.position.x ? 1 : -1
    citizen.group.position.x += citizen.direction * citizen.speed * dt
    citizen.group.position.z += Math.sin(performance.now() * .002 + citizen.phase) * dt * .35
    citizen.group.rotation.y = citizen.direction > 0 ? Math.PI / 2 : -Math.PI / 2
    citizen.phase += dt * citizen.speed * 4
    citizen.legA.rotation.z = Math.sin(citizen.phase) * .65
    citizen.legB.rotation.z = -Math.sin(citizen.phase) * .65
    citizen.head.rotation.z = distance < 9 ? Math.sin(citizen.phase * .5) * .2 : 0
    if (citizen.group.position.x > 63) citizen.group.position.x = -63
    if (citizen.group.position.x < -63) citizen.group.position.x = 63
  }
}

function updateCars(dt) {
  for (const car of cars) {
    if (!car.wrecked) {
      car.group.position.x += car.speed * car.direction * dt
      car.group.rotation.y = car.direction > 0 ? 0 : Math.PI
      if (car.group.position.x > 64) car.group.position.x = -64
      if (car.group.position.x < -64) car.group.position.x = 64
      continue
    }

    car.wreckTime += dt
    car.velocity.y -= 13 * dt
    car.group.position.addScaledVector(car.velocity, dt)
    car.group.rotation.x += car.spin.x * dt
    car.group.rotation.y += car.spin.y * dt
    car.group.rotation.z += car.spin.z * dt
    if (car.wreckTime < 1.25 && Math.random() < dt * 18) {
      const fire = car.group.position.clone().add(new THREE.Vector3((Math.random() - .5) * 2, 1, Math.random() - .5))
      spawnParticle(fire, Math.random() > .45 ? palette.orange : palette.yellow, new THREE.Vector3(0, 2 + Math.random() * 2, 0), .25 + Math.random() * .35, .5)
    }
    if (car.group.position.y < .35) {
      car.group.position.y = .35
      car.velocity.y *= -.28
      car.velocity.x *= .68
      car.velocity.z *= .68
      car.spin.multiplyScalar(.75)
    }
  }
}

function updateTrees(dt) {
  for (const tree of trees) {
    if (tree.alive) continue
    tree.velocity.y -= 11 * dt
    tree.group.position.addScaledVector(tree.velocity, dt)
    tree.group.rotation.z += tree.spin * dt
    if (tree.group.position.y < 0) {
      tree.group.position.y = 0
      tree.velocity.y *= -.2
      tree.velocity.x *= .68
      tree.velocity.z *= .68
      tree.spin *= .78
    }
  }
}

function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i -= 1) {
    const particle = particles[i]
    particle.life -= dt
    particle.velocity.y -= particle.gravity * dt
    particle.mesh.position.addScaledVector(particle.velocity, dt)
    particle.mesh.rotation.x += dt * 5
    particle.mesh.rotation.z += dt * 4
    if (particle.grow) {
      const growth = 1 + dt * 5
      particle.mesh.scale.multiplyScalar(growth)
      particle.mesh.material.opacity = Math.max(0, particle.life / particle.maxLife)
    }
    if (particle.mesh.position.y < .1 && particle.gravity > 0) {
      particle.mesh.position.y = .1
      particle.velocity.y *= -.32
      particle.velocity.x *= .72
      particle.velocity.z *= .72
    }
    if (particle.life <= 0) {
      scene.remove(particle.mesh)
      if (particle.grow) particle.mesh.material.dispose()
      particle.mesh.geometry.dispose()
      particles.splice(i, 1)
    }
  }
}

function updateScenery(dt) {
  for (const cloud of clouds) {
    cloud.position.x += cloud.userData.speed * dt
    if (cloud.position.x > 68) cloud.position.x = -68
  }

  const train = scenery.find(item => item.userData.speed && item.position.z < -10)
  if (train) {
    train.position.x += train.userData.speed * dt
    if (train.position.x > 78) train.position.x = -68
  }
}

function updateCamera(dt) {
  const targetPosition = new THREE.Vector3(player.position.x + 11, Math.max(18, player.position.y + 14), player.position.z + 27)
  const smoothing = 1 - Math.pow(.002, dt)
  camera.position.lerp(targetPosition, smoothing)
  const lookTarget = new THREE.Vector3(player.position.x + player.facing * 3, Math.max(4, player.position.y + 3), player.position.z - 2)
  camera.lookAt(lookTarget)
}

function resize() {
  const width = window.innerWidth
  const height = window.innerHeight
  const scale = width < 700 ? 2 : 3
  renderer.setSize(Math.max(320, Math.floor(width / scale)), Math.max(200, Math.floor(height / scale)), false)
  camera.aspect = width / height
  camera.updateProjectionMatrix()
  const heroWidth = Math.max(280, heroCanvas.clientWidth)
  const heroHeight = Math.max(360, heroCanvas.clientHeight)
  heroRenderer.setSize(Math.floor(heroWidth / 2), Math.floor(heroHeight / 2), false)
  heroCamera.aspect = heroWidth / heroHeight
  heroCamera.updateProjectionMatrix()
}

let started = false
let paused = true

function startGame() {
  initAudio()
  started = true
  paused = false
  setBanjoVolume()
  startScreen.classList.add('leaving')
  pauseScreen.hidden = true
  announce('UNLEASHED!')
  tone(110, .16, 'square', .08, 180)
  setTimeout(() => tone(220, .2, 'square', .07, 220), 120)
  clock.getDelta()
}

function togglePause(forceResume = false) {
  if (!started) return
  paused = forceResume ? false : !paused
  pauseScreen.hidden = !paused
  setBanjoVolume()
  if (!paused) clock.getDelta()
}

startButton.addEventListener('click', startGame)
resumeButton.addEventListener('click', () => togglePause(true))
musicToggle.addEventListener('click', () => {
  musicEnabled = !musicEnabled
  musicToggle.setAttribute('aria-pressed', String(musicEnabled))
  musicState.textContent = musicEnabled ? 'ON' : 'OFF'
  if (started) {
    initAudio()
    setBanjoVolume()
  }
})

window.addEventListener('keydown', event => {
  const blocked = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space']
  if (blocked.includes(event.code)) event.preventDefault()
  if (!started && (event.code === 'Enter' || event.code === 'Space')) {
    startGame()
    return
  }
  if (event.code === 'Escape') {
    togglePause()
    return
  }
  if (paused || event.repeat) {
    keys.add(event.code)
    return
  }
  if (event.code === 'Space') jump()
  if (event.code === 'KeyW') doKick()
  if (event.code === 'KeyR') doBurp()
  keys.add(event.code)
})

window.addEventListener('keyup', event => keys.delete(event.code))
window.addEventListener('blur', () => {
  keys.clear()
  if (started && !paused) togglePause()
})
window.addEventListener('resize', resize)

createGround()
createSkyline()
createStreetDetails()
createHighline()
populateCity()
populateCitizens()
populateCars()
resize()
updateHud()
camera.position.set(player.position.x + 11, 18, player.position.z + 27)

function loop() {
  requestAnimationFrame(loop)
  const dt = Math.min(clock.getDelta(), .04)
  if (!paused) {
    updatePlayer(dt)
    updateBuildings(dt)
    updateCitizens(dt)
    updateCars(dt)
    updateTrees(dt)
    updateParticles(dt)
    updateScenery(dt)
    updateCamera(dt)
    updateBanjo()
    announcementTimer = Math.max(0, announcementTimer - dt)
  }
  const heroTime = performance.now() * .001
  heroGoat.position.y = -1.2 + Math.abs(Math.sin(heroTime * 5)) * .12
  heroGoat.rotation.z = Math.sin(heroTime * 3) * .025
  heroLasers.scale.z = .95 + Math.sin(heroTime * 24) * .05
  heroRenderer.render(heroScene, heroCamera)
  renderer.render(scene, camera)
}

loop()
