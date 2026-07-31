import * as THREE from './assets/vendor/three.module.min.js'
import { GLTFLoader } from './assets/vendor/GLTFLoader.js'

const canvas = document.querySelector('#game')
const briefing = document.querySelector('#briefing')
const result = document.querySelector('#result')
const deploy = document.querySelector('#deploy')
const redeploy = document.querySelector('#redeploy')
const hud = document.querySelector('#hud')
const mobileControls = document.querySelector('#mobile-controls')
const healthValue = document.querySelector('#health-value')
const healthBar = document.querySelector('#health-bar')
const ammoValue = document.querySelector('#ammo')
const reserveValue = document.querySelector('#reserve')
const weaponName = document.querySelector('#weapon-name')
const timerValue = document.querySelector('#timer')
const killValue = document.querySelector('#kills')
const stanceValue = document.querySelector('#stance')
const fpsValue = document.querySelector('#fps')
const pickupPrompt = document.querySelector('#pickup-prompt')
const pickupName = document.querySelector('#pickup-name')
const hitMarker = document.querySelector('#hit-marker')
const damageFlash = document.querySelector('#damage-flash')
const killFeed = document.querySelector('#kill-feed')
const coarsePointer = matchMedia('(hover: none), (pointer: coarse)').matches

const scene = new THREE.Scene()
scene.background = new THREE.Color(0xb4a27f)
scene.fog = new THREE.FogExp2(0xb3a37f, 0.017)

const camera = new THREE.PerspectiveCamera(72, innerWidth / innerHeight, 0.05, 100)
camera.rotation.order = 'YXZ'
camera.position.set(0, 1.7, 15)
scene.add(camera)

const renderer = new THREE.WebGLRenderer({ canvas, antialias: !coarsePointer, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(devicePixelRatio, coarsePointer ? 1.15 : 1.65))
renderer.setSize(innerWidth, innerHeight)
renderer.shadowMap.enabled = !coarsePointer
renderer.shadowMap.type = THREE.PCFSoftShadowMap
renderer.outputColorSpace = THREE.SRGBColorSpace
renderer.toneMapping = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure = 1.08

const textureLoader = new THREE.TextureLoader()

function loadTexture(path, x, y) {
  const texture = textureLoader.load(path)
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping
  texture.repeat.set(x, y)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy())
  return texture
}

const asphaltTexture = loadTexture('./assets/textures/asphalt.jpg', 18, 18)
const pavementTexture = loadTexture('./assets/textures/pavement.jpg', 10, 2)
const concreteTexture = loadTexture('./assets/textures/concrete.jpg', 3, 2)
const brickTexture = loadTexture('./assets/textures/factory-brick.jpg', 5, 2)
const metalTexture = loadTexture('./assets/textures/corrugated-iron.jpg', 4, 2)
const asphaltMaterial = new THREE.MeshStandardMaterial({ map: asphaltTexture, color: 0x797268, roughness: .98, metalness: 0 })
const pavementMaterial = new THREE.MeshStandardMaterial({ map: pavementTexture, color: 0xaaa191, roughness: .94, metalness: .01 })
const concreteMaterial = new THREE.MeshStandardMaterial({ map: concreteTexture, color: 0xaaa28f, roughness: .92, metalness: .03 })
const brickMaterial = new THREE.MeshStandardMaterial({ map: brickTexture, color: 0x8a7566, roughness: .94, metalness: 0 })
const metalMaterial = new THREE.MeshStandardMaterial({ map: metalTexture, color: 0x7a786d, roughness: .72, metalness: .42 })
const steelMaterial = new THREE.MeshStandardMaterial({ color: 0x565b58, roughness: .58, metalness: .62 })
const rustMaterial = new THREE.MeshStandardMaterial({ color: 0x6f4636, roughness: .88, metalness: .18 })
const darkMaterial = new THREE.MeshStandardMaterial({ color: 0x222724, roughness: .72, metalness: .3 })
const glassMaterial = new THREE.MeshStandardMaterial({ color: 0x17252a, roughness: .22, metalness: .5, emissive: 0x071217, emissiveIntensity: .35 })

scene.add(new THREE.HemisphereLight(0xdbe5e6, 0x383932, 2.35))
const sun = new THREE.DirectionalLight(0xffd9a2, 4.1)
sun.position.set(-20, 32, 14)
sun.castShadow = !coarsePointer
sun.shadow.mapSize.set(1024, 1024)
sun.shadow.camera.left = -42
sun.shadow.camera.right = 42
sun.shadow.camera.top = 42
sun.shadow.camera.bottom = -42
sun.shadow.bias = -.001
scene.add(sun)

const ground = new THREE.Mesh(new THREE.PlaneGeometry(84, 84), asphaltMaterial)
ground.rotation.x = -Math.PI / 2
ground.receiveShadow = true
scene.add(ground)

const obstacleMeshes = []
const collisionBoxes = []
const decorations = []

function addBox(x, y, z, width, height, depth, material = concreteMaterial, collision = true) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material)
  mesh.position.set(x, y, z)
  mesh.castShadow = !coarsePointer
  mesh.receiveShadow = true
  scene.add(mesh)
  if (collision) {
    obstacleMeshes.push(mesh)
    collisionBoxes.push({ x, z, hx: width / 2, hz: depth / 2 })
  }
  return mesh
}

function addBoundary(x, z, width, depth) {
  collisionBoxes.push({ x, z, hx: width / 2, hz: depth / 2 })
}

addBoundary(0, -41, 82, 2)
addBoundary(0, 41, 82, 2)
addBoundary(-41, 0, 2, 82)
addBoundary(41, 0, 2, 82)

addBox(-21, .09, 0, 16, .18, 78, pavementMaterial, false)
addBox(21, .09, 0, 16, .18, 78, pavementMaterial, false)
addBox(0, .1, -21, 26, .2, 12, pavementMaterial, false)
addBox(0, .1, 21, 26, .2, 12, pavementMaterial, false)

for (let z = -34; z <= 34; z += 5) addBox(0, .025, z, .16, .05, 2.7, new THREE.MeshBasicMaterial({ color: 0xc1a35d }), false)
for (let x = -35; x <= 35; x += 5) addBox(x, .026, 0, 2.7, .052, .13, new THREE.MeshBasicMaterial({ color: 0xd4cfc0 }), false)

const loader = new GLTFLoader()
const assetModels = {}
const assetAnimations = {}
let assetsReady = false
let cityBuilt = false
deploy.disabled = true
deploy.textContent = 'LOADING CITY'

const modelPaths = {
  pistol: './assets/models/blaster-n.glb',
  rifle: './assets/models/blaster-e.glb',
  shotgun: './assets/models/blaster-p.glb',
  crate: './assets/models/crate-medium.glb',
  roadStraight: './assets/roads/road-straight.glb',
  roadCross: './assets/roads/road-crossroad-path.glb',
  streetLight: './assets/roads/light-curved.glb',
  streetLightDouble: './assets/roads/light-curved-double.glb',
  barrier: './assets/roads/construction-barrier.glb',
  cone: './assets/roads/construction-cone.glb',
  roadSign: './assets/roads/sign-highway.glb',
  crane: './assets/factory/crane.glb',
  hopper: './assets/factory/hopper-high-round.glb',
  machine: './assets/factory/machine-fortified.glb',
  pipeBend: './assets/factory/pipe-large-bend.glb',
  pipeLong: './assets/factory/pipe-large-long.glb',
  pipeValve: './assets/factory/pipe-large-valve.glb',
  catwalk: './assets/factory/catwalk-straight.glb',
  stairs: './assets/factory/catwalk-stairs.glb',
  factoryBox: './assets/factory/box-large.glb',
  factoryBoxWide: './assets/factory/box-wide.glb',
  robotArm: './assets/factory/robot-arm-a.glb',
  trafficWarning: './assets/factory/warning-traffic.glb',
  towerA: './assets/buildings/building-sample-tower-a.glb',
  towerB: './assets/buildings/building-sample-tower-b.glb',
  towerC: './assets/buildings/building-sample-tower-c.glb',
  towerD: './assets/buildings/building-sample-tower-d.glb',
  houseA: './assets/buildings/building-sample-house-a.glb',
  houseB: './assets/buildings/building-sample-house-b.glb',
  houseC: './assets/buildings/building-sample-house-c.glb',
  acA: './assets/buildings/detail-ac-a.glb',
  acB: './assets/buildings/detail-ac-b.glb'
}

for (const letter of 'abcdefghijklmnopqrst') modelPaths[`city${letter.toUpperCase()}`] = `./assets/city/building-${letter}.glb`
for (const letter of 'abcde') modelPaths[`enemy${letter.toUpperCase()}`] = `./assets/characters/character-${letter}.glb`

function styleAsset(model, color, roughness, metalness) {
  model.traverse(node => {
    if (!node.isMesh) return
    node.material = node.material.clone()
    node.material.color.multiply(new THREE.Color(color))
    node.material.roughness = roughness
    node.material.metalness = metalness
  })
}

function prepareAsset(model, shadows = true) {
  model.traverse(node => {
    if (!node.isMesh) return
    node.castShadow = shadows && !coarsePointer
    node.receiveShadow = true
  })
}

function placeAsset(key, x, z, scale, rotation = 0, collision = false, tint = null) {
  if (!assetModels[key]) return null
  const model = assetModels[key].clone(true)
  if (tint) styleAsset(model, tint, .7, .18)
  prepareAsset(model, collision)
  model.scale.setScalar(scale)
  model.rotation.y = rotation
  model.position.set(x, 0, z)
  scene.add(model)
  scene.updateMatrixWorld(true)
  let box = new THREE.Box3().setFromObject(model)
  model.position.y -= box.min.y
  scene.updateMatrixWorld(true)
  box = new THREE.Box3().setFromObject(model)
  if (collision) {
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    collisionBoxes.push({ x: center.x, z: center.z, hx: Math.max(.3, size.x * .44), hz: Math.max(.3, size.z * .44) })
    model.traverse(node => { if (node.isMesh) obstacleMeshes.push(node) })
  }
  decorations.push(model)
  return model
}

function addWarehouse(x, z, width, depth, height, material) {
  addBox(x, height / 2, z, width, height, depth, material)
  addBox(x, height + .18, z, width + .18, .36, depth + .18, metalMaterial, false)
  const door = addBox(x, 1.65, z + depth / 2 + .012, 3.2, 3.3, .08, darkMaterial, false)
  door.castShadow = false
  for (let offset = -width * .32; offset <= width * .32; offset += width * .32) addBox(x + offset, height * .62, z + depth / 2 + .02, 1.2, 1.1, .1, glassMaterial, false)
}

function buildCity() {
  if (cityBuilt) return
  cityBuilt = true
  for (let z = -36; z <= 36; z += 8) {
    if (Math.abs(z) < 5) continue
    placeAsset('roadStraight', 0, z, 8, 0, false)
  }
  for (let x = -36; x <= 36; x += 8) {
    if (Math.abs(x) < 5) continue
    placeAsset('roadStraight', x, 0, 8, Math.PI / 2, false)
  }
  placeAsset('roadCross', 0, 0, 8, 0, false)

  const cityLayout = [
    ['cityA', -33, -31, 5.8, 0], ['cityB', -20, -33, 5.6, 0], ['cityC', -7, -34, 5.8, 0], ['cityD', 7, -34, 5.8, 0], ['cityE', 21, -33, 5.8, 0], ['cityF', 34, -28, 5.8, Math.PI / 2],
    ['cityG', -34, -17, 5.6, -Math.PI / 2], ['cityH', -35, -4, 5.8, -Math.PI / 2], ['cityI', -35, 11, 5.8, -Math.PI / 2], ['cityJ', -33, 27, 5.7, Math.PI],
    ['cityK', -21, 34, 5.8, Math.PI], ['cityL', -8, 35, 5.8, Math.PI], ['cityM', 7, 35, 5.7, Math.PI], ['cityN', 21, 34, 5.7, Math.PI], ['cityO', 34, 28, 5.8, Math.PI / 2],
    ['cityP', 35, 14, 5.8, Math.PI / 2], ['cityQ', 35, 1, 5.8, Math.PI / 2], ['cityR', 35, -14, 5.8, Math.PI / 2]
  ]
  cityLayout.forEach(args => placeAsset(...args, true))

  placeAsset('towerA', -20, 13, 4.4, Math.PI / 2, true)
  placeAsset('towerB', 21, 14, 4.3, -Math.PI / 2, true)
  placeAsset('houseA', -19, -14, 4.8, Math.PI / 2, true)
  placeAsset('houseB', 20, -14, 4.7, -Math.PI / 2, true)
  addWarehouse(-12, 15, 10, 12, 6.4, brickMaterial)
  addWarehouse(13, -15, 11, 11, 5.8, metalMaterial)

  const lights = [[-6, -27, 0], [6, -27, Math.PI], [-6, -12, 0], [6, -12, Math.PI], [-6, 13, 0], [6, 13, Math.PI], [-6, 28, 0], [6, 28, Math.PI], [-27, -6, Math.PI / 2], [-27, 6, -Math.PI / 2], [27, -6, Math.PI / 2], [27, 6, -Math.PI / 2]]
  lights.forEach(([x, z, rotation]) => placeAsset('streetLight', x, z, 6.2, rotation, false))
  placeAsset('streetLightDouble', 0, -18, 6.2, 0, false)
  placeAsset('streetLightDouble', 0, 18, 6.2, Math.PI, false)
  placeAsset('roadSign', 1, -29, 5.5, Math.PI, false)

  ;[[-4, 7, 0], [4, 7, Math.PI], [-5, -7, 0], [5, -7, Math.PI]].forEach(args => placeAsset('barrier', ...args, 3.2, args[2], true))
  ;[[-6, 6], [-5, 6], [5, -6], [6, -6], [-2, -19], [2, 19]].forEach(([x, z], index) => placeAsset('cone', x, z, 2.2, index, false))

  placeAsset('crane', -24, 22, 2.25, Math.PI / 2, true)
  placeAsset('hopper', 24, 23, 2.5, 0, true)
  placeAsset('machine', 17, -22, 2.8, Math.PI, true)
  placeAsset('robotArm', 24, -23, 2.5, -.5, true)
  placeAsset('catwalk', -12, 21, 3.1, Math.PI / 2, true)
  placeAsset('stairs', -8, 22, 3.1, Math.PI / 2, true)
  placeAsset('pipeLong', 12, 21, 2.6, Math.PI / 2, true)
  placeAsset('pipeValve', 16, 21, 2.6, Math.PI / 2, true)
  placeAsset('pipeBend', 20, 21, 2.6, Math.PI / 2, true)
  placeAsset('trafficWarning', 8, -20, 2.7, 0, false)

  const crateLayout = [[-3, 10, 0], [-2, 10, 0], [3, -10, 0], [4, -10, 0], [-25, 8, .4], [25, -7, -.4], [10, 25, 0], [-10, -24, 0]]
  crateLayout.forEach(([x, z, rotation], index) => placeAsset(index % 3 ? 'factoryBox' : 'factoryBoxWide', x, z, 2.2, rotation, true))
}

async function loadAssets() {
  try {
    await Promise.all(Object.entries(modelPaths).map(async ([key, path]) => {
      const data = await loader.loadAsync(path)
      assetModels[key] = data.scene
      assetAnimations[key] = data.animations
    }))
    buildCity()
    bots.forEach((bot, index) => bot.setModel(`enemy${'ABCDE'[index]}`))
    pickups.forEach(item => addPickupModel(item))
    showWeaponModel()
    assetsReady = true
    deploy.disabled = false
    deploy.textContent = 'DEPLOY'
  } catch (error) {
    deploy.textContent = 'ASSET LOAD FAILED'
    console.error(error)
  }
}

loadAssets()

const weaponDefinitions = {
  pistol: { label: 'VX-9 SIDEARM', pickup: 'VX-9 SIDEARM', mag: 12, reserve: 60, damage: 34, rate: .22, reload: 1.15, spread: .004, pellets: 1, scale: .24 },
  rifle: { label: 'AR-4 CARBINE', pickup: 'AR-4 CARBINE', mag: 30, reserve: 120, damage: 22, rate: .095, reload: 1.65, spread: .009, pellets: 1, scale: .21 },
  shotgun: { label: 'M12 BREACHER', pickup: 'M12 BREACHER', mag: 6, reserve: 30, damage: 17, rate: .7, reload: 1.8, spread: .044, pellets: 7, scale: .2 }
}

const inventory = {
  pistol: { owned: true, ammo: 12, reserve: 60 },
  rifle: { owned: false, ammo: 30, reserve: 120 },
  shotgun: { owned: false, ammo: 6, reserve: 30 }
}

const weaponRig = new THREE.Group()
weaponRig.position.set(.3, -.27, -.54)
camera.add(weaponRig)
let weaponView = null
let currentWeaponKey = 'pistol'
let recoil = 0
let muzzleLife = 0
const muzzle = new THREE.PointLight(0xff9c3a, 0, 2.5)
muzzle.position.set(.05, .02, -.65)
camera.add(muzzle)

function fallbackWeapon() {
  const group = new THREE.Group()
  const body = new THREE.Mesh(new THREE.BoxGeometry(.13, .15, .55), darkMaterial)
  body.position.z = -.18
  const grip = new THREE.Mesh(new THREE.BoxGeometry(.11, .28, .13), darkMaterial)
  grip.position.set(0, -.16, .01)
  grip.rotation.x = -.2
  group.add(body, grip)
  return group
}

function showWeaponModel() {
  if (weaponView) weaponRig.remove(weaponView)
  weaponView = assetModels[currentWeaponKey] ? assetModels[currentWeaponKey].clone(true) : fallbackWeapon()
  if (assetModels[currentWeaponKey]) styleAsset(weaponView, 0x777970, .52, .3)
  const definition = weaponDefinitions[currentWeaponKey]
  weaponView.scale.setScalar(definition.scale)
  weaponView.rotation.set(-.12, Math.PI / 2, 0)
  weaponView.position.set(0, -.06, -.08)
  weaponRig.add(weaponView)
  updateHud()
}

const pickups = [
  { key: 'rifle', x: 17, z: -18, baseY: .52, model: null, active: true },
  { key: 'shotgun', x: -20, z: 8, baseY: .52, model: null, active: true }
]

pickups.forEach(item => addPickupModel(item))

function addPickupModel(item) {
  if (item.model) scene.remove(item.model)
  const group = new THREE.Group()
  const model = assetModels[item.key] ? assetModels[item.key].clone(true) : fallbackWeapon()
  if (assetModels[item.key]) styleAsset(model, 0x777970, .58, .26)
  model.scale.setScalar(assetModels[item.key] ? .42 : 1.2)
  model.rotation.y = Math.PI / 2
  group.add(model)
  const ring = new THREE.Mesh(new THREE.RingGeometry(.52, .62, 24), new THREE.MeshBasicMaterial({ color: 0xe8a848, transparent: true, opacity: .55, side: THREE.DoubleSide }))
  ring.rotation.x = -Math.PI / 2
  ring.position.y = -.45
  group.add(ring)
  group.position.set(item.x, item.baseY, item.z)
  scene.add(group)
  item.model = group
}

const botMaterial = new THREE.MeshStandardMaterial({ color: 0x43493f, roughness: .78, metalness: .08 })
const botArmorMaterial = new THREE.MeshStandardMaterial({ color: 0x2b302c, roughness: .7, metalness: .22 })
const botAccentMaterial = new THREE.MeshStandardMaterial({ color: 0x9f4a35, roughness: .68, metalness: .18 })
const botHitMeshes = []
const botSpawns = [[-22, -14], [21, -18], [20, 14], [-20, 20], [1, -21], [23, 1], [-21, 1]]
const bots = []

class Bot {
  constructor(index) {
    this.index = index
    this.group = new THREE.Group()
    const legs = new THREE.Mesh(new THREE.CylinderGeometry(.32, .38, .8, 7), botMaterial)
    legs.position.y = .48
    const torso = new THREE.Mesh(new THREE.CylinderGeometry(.42, .34, .82, 7), botArmorMaterial)
    torso.position.y = 1.18
    const vest = new THREE.Mesh(new THREE.BoxGeometry(.62, .48, .28), botAccentMaterial)
    vest.position.set(0, 1.22, -.27)
    const head = new THREE.Mesh(new THREE.SphereGeometry(.25, 10, 8), botMaterial)
    head.position.y = 1.78
    const visor = new THREE.Mesh(new THREE.BoxGeometry(.34, .1, .12), new THREE.MeshStandardMaterial({ color: 0x171b18, metalness: .75, roughness: .26 }))
    visor.position.set(0, 1.82, -.21)
    const gun = new THREE.Mesh(new THREE.BoxGeometry(.12, .13, .68), darkMaterial)
    gun.position.set(.34, 1.25, -.35)
    gun.rotation.x = -.08
    this.group.add(legs, torso, vest, head, visor, gun)
    ;[legs, torso, vest, head, visor].forEach(mesh => {
      mesh.castShadow = !coarsePointer
      mesh.userData.bot = this
      botHitMeshes.push(mesh)
    })
    scene.add(this.group)
    this.health = 100
    this.alive = true
    this.cooldown = 1 + Math.random()
    this.strafe = Math.random() > .5 ? 1 : -1
    this.respawnAt = 0
    this.place(index)
  }

  place(offset = 0) {
    const spawn = botSpawns[(this.index + offset) % botSpawns.length]
    this.group.position.set(spawn[0] + Math.random() * 2 - 1, 0, spawn[1] + Math.random() * 2 - 1)
    this.group.visible = true
    this.health = 100
    this.alive = true
    this.cooldown = .7 + Math.random() * 1.2
  }

  update(dt, elapsed) {
    if (!this.alive) {
      if (running && elapsed >= this.respawnAt && kills < targetKills) this.place(Math.floor(elapsed) + 1)
      return
    }
    const dx = player.x - this.group.position.x
    const dz = player.z - this.group.position.z
    const distance = Math.hypot(dx, dz)
    const visible = distance < 25 && !segmentBlocked(this.group.position.x, this.group.position.z, player.x, player.z)
    if (visible) {
      const targetAngle = Math.atan2(dx, dz)
      this.group.rotation.y = approachAngle(this.group.rotation.y, targetAngle, dt * 4.5)
      if (distance > 8) this.move(dx / distance * dt * 1.25, dz / distance * dt * 1.25)
      if (distance < 6) this.move(-dx / distance * dt * .8, -dz / distance * dt * .8)
      this.move(dz / distance * this.strafe * dt * .42, -dx / distance * this.strafe * dt * .42)
      this.cooldown -= dt
      if (this.cooldown <= 0) {
        this.shoot(distance)
        this.cooldown = .72 + Math.random() * .7
        if (Math.random() < .3) this.strafe *= -1
      }
    } else {
      const angle = elapsed * .18 + this.index * 1.37
      this.group.rotation.y = approachAngle(this.group.rotation.y, angle, dt * 1.5)
      this.move(Math.sin(angle) * dt * .42, Math.cos(angle) * dt * .42)
    }
  }

  move(dx, dz) {
    const nextX = this.group.position.x + dx
    const nextZ = this.group.position.z + dz
    if (canMove(nextX, this.group.position.z, .5)) this.group.position.x = nextX
    if (canMove(this.group.position.x, nextZ, .5)) this.group.position.z = nextZ
  }

  shoot(distance) {
    const start = new THREE.Vector3(this.group.position.x, 1.3, this.group.position.z)
    const end = new THREE.Vector3(player.x, 1.45, player.z)
    const accuracy = Math.max(.28, .82 - distance * .018)
    const hit = Math.random() < accuracy
    if (!hit) {
      end.x += (Math.random() - .5) * distance * .16
      end.z += (Math.random() - .5) * distance * .16
      end.y += (Math.random() - .5) * 2
    }
    addTracer(start, end, 0xd95a3e)
    botShotSound(distance)
    if (hit) damagePlayer(7 + Math.floor(Math.random() * 7))
  }

  hit(damage, point) {
    if (!this.alive) return
    this.health -= damage
    makeImpact(point)
    if (this.health <= 0) {
      this.alive = false
      this.group.visible = false
      this.respawnAt = matchElapsed + 3.2
      kills += 1
      killValue.textContent = kills
      addFeed('YOU', `HOSTILE ${String(this.index + 1).padStart(2, '0')}`)
      deathSound()
      if (kills >= targetKills) endMatch(true)
    }
  }
}

for (let i = 0; i < 5; i += 1) bots.push(new Bot(i))

const player = { x: 0, z: 15, yaw: 0, pitch: 0, health: 100, radius: .42 }
const keys = new Set()
const targetKills = 12
let running = false
let started = false
let matchElapsed = 0
let kills = 0
let shots = 0
let hitShots = 0
let nextShot = 0
let fireHeld = false
let reloadLeft = 0
let currentPickup = null
let audioContext = null
let lastFrame = performance.now()
let fpsFrames = 0
let fpsElapsed = 0
let bobTime = 0
let moveAmount = 0
const tracers = []
const impacts = []
const raycaster = new THREE.Raycaster()

showWeaponModel()

function canMove(x, z, radius = player.radius) {
  if (x < -28.8 || x > 28.8 || z < -28.8 || z > 28.8) return false
  return !collisionBoxes.some(box => x + radius > box.x - box.hx && x - radius < box.x + box.hx && z + radius > box.z - box.hz && z - radius < box.z + box.hz)
}

function segmentBlocked(x1, z1, x2, z2) {
  const dx = x2 - x1
  const dz = z2 - z1
  const length = Math.hypot(dx, dz)
  const steps = Math.ceil(length * 1.4)
  for (let i = 1; i < steps; i += 1) {
    const x = x1 + dx * i / steps
    const z = z1 + dz * i / steps
    if (collisionBoxes.some(box => x > box.x - box.hx && x < box.x + box.hx && z > box.z - box.hz && z < box.z + box.hz)) return true
  }
  return false
}

function approachAngle(current, target, amount) {
  let delta = (target - current + Math.PI) % (Math.PI * 2) - Math.PI
  if (delta < -Math.PI) delta += Math.PI * 2
  return current + Math.max(-amount, Math.min(amount, delta))
}

function updatePlayer(dt) {
  let forward = 0
  let side = 0
  if (keys.has('KeyW')) forward += 1
  if (keys.has('KeyS')) forward -= 1
  if (keys.has('KeyD')) side += 1
  if (keys.has('KeyA')) side -= 1
  forward += mobileMove.y
  side += mobileMove.x
  const magnitude = Math.hypot(forward, side)
  if (magnitude > 1) {
    forward /= magnitude
    side /= magnitude
  }
  const sprinting = (keys.has('ShiftLeft') || keys.has('ShiftRight') || mobileSprint) && forward > .2
  const speed = sprinting ? 6.8 : 4.15
  const sin = Math.sin(player.yaw)
  const cos = Math.cos(player.yaw)
  const dx = (side * cos - forward * sin) * speed * dt
  const dz = (-side * sin - forward * cos) * speed * dt
  if (canMove(player.x + dx, player.z)) player.x += dx
  if (canMove(player.x, player.z + dz)) player.z += dz
  moveAmount += (magnitude - moveAmount) * Math.min(1, dt * 12)
  if (magnitude > .08) bobTime += dt * (sprinting ? 12 : 8)
  const bob = Math.sin(bobTime) * .025 * moveAmount
  camera.position.set(player.x, 1.68 + Math.abs(Math.cos(bobTime * .5)) * .024 * moveAmount, player.z)
  camera.rotation.y = player.yaw
  camera.rotation.x = player.pitch + bob * .12
  weaponRig.position.x = .3 + Math.cos(bobTime * .5) * .018 * moveAmount
  weaponRig.position.y = -.27 + Math.sin(bobTime) * .018 * moveAmount - recoil * .055
  weaponRig.position.z = -.54 + recoil * .12
  weaponRig.rotation.x = recoil * .09
  recoil = Math.max(0, recoil - dt * 7)
  stanceValue.textContent = sprinting ? 'SPRINT' : magnitude > .08 ? 'WALK' : 'HOLD'
}

function shoot() {
  if (!running || reloadLeft > 0 || matchElapsed < nextShot) return
  const definition = weaponDefinitions[currentWeaponKey]
  const state = inventory[currentWeaponKey]
  if (state.ammo <= 0) {
    drySound()
    nextShot = matchElapsed + .22
    return
  }
  state.ammo -= 1
  shots += 1
  nextShot = matchElapsed + definition.rate
  recoil = Math.min(1.8, recoil + (currentWeaponKey === 'shotgun' ? 1.25 : .55))
  muzzle.intensity = currentWeaponKey === 'shotgun' ? 18 : 10
  muzzleLife = .045
  gunSound(currentWeaponKey)
  let triggerHit = false
  for (let pellet = 0; pellet < definition.pellets; pellet += 1) {
    const direction = new THREE.Vector3()
    camera.getWorldDirection(direction)
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion)
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion)
    direction.addScaledVector(right, (Math.random() - .5) * definition.spread)
    direction.addScaledVector(up, (Math.random() - .5) * definition.spread)
    direction.normalize()
    raycaster.set(camera.position, direction)
    raycaster.far = 45
    const botHits = raycaster.intersectObjects(botHitMeshes, false).filter(hit => hit.object.userData.bot.alive)
    const wallHits = raycaster.intersectObjects(obstacleMeshes, false)
    const botHit = botHits[0]
    const wallHit = wallHits[0]
    let end = camera.position.clone().addScaledVector(direction, 38)
    if (wallHit) end = wallHit.point
    if (botHit && (!wallHit || botHit.distance < wallHit.distance)) {
      end = botHit.point
      botHit.object.userData.bot.hit(definition.damage, botHit.point)
      triggerHit = true
    } else if (wallHit) makeImpact(wallHit.point)
    if (pellet < 3) addTracer(camera.position.clone().addScaledVector(direction, .7), end, 0xffc866)
  }
  if (triggerHit) {
    hitShots += 1
    hitMarker.classList.remove('show')
    void hitMarker.offsetWidth
    hitMarker.classList.add('show')
    hitSound()
  }
  updateHud()
}

function reload() {
  if (!running || reloadLeft > 0) return
  const definition = weaponDefinitions[currentWeaponKey]
  const state = inventory[currentWeaponKey]
  if (state.ammo >= definition.mag || state.reserve <= 0) return
  reloadLeft = definition.reload
  reloadSound()
  weaponName.textContent = 'RELOADING'
}

function finishReload() {
  const definition = weaponDefinitions[currentWeaponKey]
  const state = inventory[currentWeaponKey]
  const needed = definition.mag - state.ammo
  const amount = Math.min(needed, state.reserve)
  state.ammo += amount
  state.reserve -= amount
  reloadLeft = 0
  updateHud()
}

function switchWeapon(key) {
  if (!inventory[key]?.owned || key === currentWeaponKey) return
  currentWeaponKey = key
  reloadLeft = 0
  showWeaponModel()
  equipSound()
}

function usePickup() {
  if (!currentPickup) return
  const item = currentPickup
  inventory[item.key].owned = true
  inventory[item.key].ammo = weaponDefinitions[item.key].mag
  item.active = false
  item.model.visible = false
  currentPickup = null
  pickupPrompt.classList.remove('visible')
  switchWeapon(item.key)
  addFeed('EQUIPPED', weaponDefinitions[item.key].pickup)
}

function updatePickups(elapsed) {
  currentPickup = null
  pickups.forEach((item, index) => {
    if (!item.active || !item.model) return
    item.model.rotation.y += .65 / 60
    item.model.position.y = item.baseY + Math.sin(elapsed * 2.2 + index) * .1
    const distance = Math.hypot(player.x - item.x, player.z - item.z)
    if (distance < 2.1) currentPickup = item
  })
  pickupPrompt.classList.toggle('visible', Boolean(currentPickup))
  if (currentPickup) pickupName.textContent = weaponDefinitions[currentPickup.key].pickup
}

function addTracer(start, end, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints([start, end])
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: .75 })
  const line = new THREE.Line(geometry, material)
  scene.add(line)
  tracers.push({ line, life: .075 })
}

function makeImpact(point) {
  const material = new THREE.MeshBasicMaterial({ color: 0xffc86b, transparent: true, opacity: .8 })
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(.045, 5, 4), material)
  mesh.position.copy(point)
  scene.add(mesh)
  impacts.push({ mesh, life: .18 })
}

function updateEffects(dt) {
  for (let i = tracers.length - 1; i >= 0; i -= 1) {
    tracers[i].life -= dt
    tracers[i].line.material.opacity = Math.max(0, tracers[i].life * 10)
    if (tracers[i].life <= 0) {
      scene.remove(tracers[i].line)
      tracers[i].line.geometry.dispose()
      tracers[i].line.material.dispose()
      tracers.splice(i, 1)
    }
  }
  for (let i = impacts.length - 1; i >= 0; i -= 1) {
    impacts[i].life -= dt
    impacts[i].mesh.scale.multiplyScalar(1.08)
    impacts[i].mesh.material.opacity = impacts[i].life * 4
    if (impacts[i].life <= 0) {
      scene.remove(impacts[i].mesh)
      impacts[i].mesh.geometry.dispose()
      impacts[i].mesh.material.dispose()
      impacts.splice(i, 1)
    }
  }
  if (muzzleLife > 0) {
    muzzleLife -= dt
    if (muzzleLife <= 0) muzzle.intensity = 0
  }
}

function damagePlayer(amount) {
  if (!running) return
  player.health = Math.max(0, player.health - amount)
  damageFlash.classList.add('show')
  setTimeout(() => damageFlash.classList.remove('show'), 90)
  hurtSound()
  updateHud()
  if (player.health <= 0) endMatch(false)
}

function updateHud() {
  const state = inventory[currentWeaponKey]
  healthValue.textContent = player.health
  healthBar.style.width = `${player.health}%`
  healthBar.style.background = player.health < 30 ? 'var(--danger)' : 'var(--amber)'
  ammoValue.textContent = String(state.ammo).padStart(2, '0')
  reserveValue.textContent = state.reserve
  weaponName.textContent = weaponDefinitions[currentWeaponKey].label
}

function addFeed(actor, target) {
  const item = document.createElement('div')
  item.className = 'feed-item'
  item.innerHTML = `<b>${actor}</b> &nbsp;—&nbsp; ${target}`
  killFeed.prepend(item)
  setTimeout(() => item.remove(), 3300)
}

function begin() {
  ensureAudio()
  if (!started || result.classList.contains('visible')) resetMatch()
  running = true
  started = true
  briefing.classList.remove('visible')
  result.classList.remove('visible')
  hud.classList.add('active')
  mobileControls.classList.toggle('active', coarsePointer)
  if (!coarsePointer) canvas.requestPointerLock()
}

function resetMatch() {
  matchElapsed = 0
  kills = 0
  shots = 0
  hitShots = 0
  player.x = 0
  player.z = 15
  player.yaw = 0
  player.pitch = 0
  player.health = 100
  currentWeaponKey = 'pistol'
  Object.entries(inventory).forEach(([key, state]) => {
    state.owned = key === 'pistol'
    state.ammo = weaponDefinitions[key].mag
    state.reserve = weaponDefinitions[key].reserve
  })
  pickups.forEach(item => {
    item.active = true
    if (item.model) item.model.visible = true
  })
  bots.forEach((bot, index) => bot.place(index))
  killValue.textContent = '0'
  killFeed.innerHTML = ''
  reloadLeft = 0
  fireHeld = false
  showWeaponModel()
  updateHud()
}

function endMatch(won) {
  if (!running) return
  running = false
  fireHeld = false
  mobileControls.classList.remove('active')
  hud.classList.remove('active')
  document.exitPointerLock?.()
  const minutes = Math.floor(matchElapsed / 60)
  const seconds = Math.floor(matchElapsed % 60)
  document.querySelector('#result-title').textContent = won ? 'SECTOR CLEAR' : 'OPERATOR DOWN'
  document.querySelector('#result-kicker').lastChild.textContent = won ? ' Mission complete' : ' Mission failed'
  document.querySelector('#result-kills').textContent = kills
  document.querySelector('#result-accuracy').textContent = `${shots ? Math.round(hitShots / shots * 100) : 0}%`
  document.querySelector('#result-time').textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
  result.classList.add('visible')
}

function ensureAudio() {
  if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)()
  if (audioContext.state === 'suspended') audioContext.resume()
}

function tone(frequency, length, gain, type = 'square', drop = 0) {
  if (!audioContext) return
  const oscillator = audioContext.createOscillator()
  const volume = audioContext.createGain()
  oscillator.type = type
  oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime)
  oscillator.frequency.exponentialRampToValueAtTime(Math.max(30, frequency - drop), audioContext.currentTime + length)
  volume.gain.setValueAtTime(gain, audioContext.currentTime)
  volume.gain.exponentialRampToValueAtTime(.001, audioContext.currentTime + length)
  oscillator.connect(volume).connect(audioContext.destination)
  oscillator.start()
  oscillator.stop(audioContext.currentTime + length)
}

function noise(length, gain) {
  if (!audioContext) return
  const count = Math.floor(audioContext.sampleRate * length)
  const buffer = audioContext.createBuffer(1, count, audioContext.sampleRate)
  const data = buffer.getChannelData(0)
  for (let i = 0; i < count; i += 1) data[i] = Math.random() * 2 - 1
  const source = audioContext.createBufferSource()
  const filter = audioContext.createBiquadFilter()
  const volume = audioContext.createGain()
  filter.type = 'lowpass'
  filter.frequency.value = 1800
  volume.gain.setValueAtTime(gain, audioContext.currentTime)
  volume.gain.exponentialRampToValueAtTime(.001, audioContext.currentTime + length)
  source.buffer = buffer
  source.connect(filter).connect(volume).connect(audioContext.destination)
  source.start()
}

function gunSound(key) {
  const heavy = key === 'shotgun'
  noise(heavy ? .25 : .11, heavy ? .22 : .12)
  tone(heavy ? 105 : key === 'rifle' ? 150 : 190, heavy ? .22 : .1, .1, 'sawtooth', heavy ? 70 : 100)
}

function botShotSound(distance) { tone(135, .1, Math.max(.018, .07 - distance * .002), 'square', 80) }
function drySound() { tone(340, .045, .035, 'square', 70) }
function reloadSound() { tone(260, .07, .035, 'triangle', 30); setTimeout(() => tone(430, .06, .028, 'triangle', 50), 180) }
function equipSound() { tone(210, .09, .04, 'triangle', 20) }
function hitSound() { tone(720, .045, .035, 'square', 150) }
function hurtSound() { tone(75, .16, .06, 'sawtooth', 30) }
function deathSound() { tone(110, .24, .05, 'sawtooth', 70) }

function update(dt) {
  if (running) {
    matchElapsed += dt
    updatePlayer(dt)
    bots.forEach(bot => bot.update(dt, matchElapsed))
    updatePickups(matchElapsed)
    if (fireHeld) shoot()
    if (reloadLeft > 0) {
      reloadLeft -= dt
      if (reloadLeft <= 0) finishReload()
    }
    const remaining = Math.max(0, 300 - matchElapsed)
    timerValue.textContent = `${String(Math.floor(remaining / 60)).padStart(2, '0')}:${String(Math.floor(remaining % 60)).padStart(2, '0')}`
    if (remaining <= 0) endMatch(false)
  }
  updateEffects(dt)
}

function loop(now) {
  const dt = Math.min(.05, (now - lastFrame) / 1000)
  lastFrame = now
  fpsFrames += 1
  fpsElapsed += dt
  if (fpsElapsed >= .5) {
    fpsValue.textContent = `${Math.round(fpsFrames / fpsElapsed)} FPS`
    fpsFrames = 0
    fpsElapsed = 0
  }
  update(dt)
  renderer.render(scene, camera)
  requestAnimationFrame(loop)
}

deploy.addEventListener('click', begin)
redeploy.addEventListener('click', begin)

document.addEventListener('keydown', event => {
  keys.add(event.code)
  if (event.code === 'KeyR') reload()
  if (event.code === 'KeyE') usePickup()
  if (event.code === 'Digit1') switchWeapon('pistol')
  if (event.code === 'Digit2') switchWeapon('rifle')
  if (event.code === 'Digit3') switchWeapon('shotgun')
})

document.addEventListener('keyup', event => keys.delete(event.code))

document.addEventListener('mousemove', event => {
  if (!running || document.pointerLockElement !== canvas) return
  player.yaw -= event.movementX * .0019
  player.pitch -= event.movementY * .0019
  player.pitch = Math.max(-1.35, Math.min(1.35, player.pitch))
})

canvas.addEventListener('mousedown', event => {
  if (!running) return
  if (document.pointerLockElement !== canvas && !coarsePointer) {
    canvas.requestPointerLock()
    return
  }
  if (event.button === 0) {
    fireHeld = true
    shoot()
  }
})

document.addEventListener('mouseup', event => {
  if (event.button === 0) fireHeld = false
})

document.addEventListener('pointerlockchange', () => {
  if (started && running && !coarsePointer && document.pointerLockElement !== canvas) {
    running = false
    deploy.textContent = 'RESUME'
    briefing.classList.add('visible')
    hud.classList.remove('active')
  }
})

let mobileMove = { x: 0, y: 0 }
let mobileSprint = false
let movePointer = null
let lookPointer = null
let lookLast = { x: 0, y: 0 }
const moveZone = document.querySelector('#move-zone')
const moveStick = document.querySelector('#move-stick i')
const lookZone = document.querySelector('#look-zone')

moveZone.addEventListener('pointerdown', event => {
  movePointer = event.pointerId
  moveZone.setPointerCapture(event.pointerId)
  updateMoveStick(event)
})

moveZone.addEventListener('pointermove', event => {
  if (event.pointerId === movePointer) updateMoveStick(event)
})

function updateMoveStick(event) {
  const rect = document.querySelector('#move-stick').getBoundingClientRect()
  const dx = event.clientX - (rect.left + rect.width / 2)
  const dy = event.clientY - (rect.top + rect.height / 2)
  const distance = Math.hypot(dx, dy)
  const limit = 35
  const scale = distance > limit ? limit / distance : 1
  mobileMove.x = dx * scale / limit
  mobileMove.y = -dy * scale / limit
  moveStick.style.transform = `translate(${dx * scale}px, ${dy * scale}px)`
}

function clearMove(event) {
  if (event.pointerId !== movePointer) return
  movePointer = null
  mobileMove = { x: 0, y: 0 }
  moveStick.style.transform = ''
}

moveZone.addEventListener('pointerup', clearMove)
moveZone.addEventListener('pointercancel', clearMove)

lookZone.addEventListener('pointerdown', event => {
  lookPointer = event.pointerId
  lookLast = { x: event.clientX, y: event.clientY }
  lookZone.setPointerCapture(event.pointerId)
})

lookZone.addEventListener('pointermove', event => {
  if (event.pointerId !== lookPointer || !running) return
  const dx = event.clientX - lookLast.x
  const dy = event.clientY - lookLast.y
  player.yaw -= dx * .0042
  player.pitch -= dy * .0035
  player.pitch = Math.max(-1.25, Math.min(1.25, player.pitch))
  lookLast = { x: event.clientX, y: event.clientY }
})

function clearLook(event) {
  if (event.pointerId === lookPointer) lookPointer = null
}

lookZone.addEventListener('pointerup', clearLook)
lookZone.addEventListener('pointercancel', clearLook)

const mobileFire = document.querySelector('#mobile-fire')
mobileFire.addEventListener('pointerdown', event => {
  event.stopPropagation()
  fireHeld = true
  shoot()
  mobileFire.setPointerCapture(event.pointerId)
})
mobileFire.addEventListener('pointerup', () => { fireHeld = false })
mobileFire.addEventListener('pointercancel', () => { fireHeld = false })

const mobileSprintButton = document.querySelector('#mobile-sprint')
mobileSprintButton.addEventListener('pointerdown', event => {
  event.stopPropagation()
  mobileSprint = true
  mobileSprintButton.setPointerCapture(event.pointerId)
})
mobileSprintButton.addEventListener('pointerup', () => { mobileSprint = false })
mobileSprintButton.addEventListener('pointercancel', () => { mobileSprint = false })
document.querySelector('#mobile-use').addEventListener('pointerdown', event => { event.stopPropagation(); usePickup() })
document.querySelector('#mobile-reload').addEventListener('pointerdown', event => { event.stopPropagation(); reload() })

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight
  camera.updateProjectionMatrix()
  renderer.setPixelRatio(Math.min(devicePixelRatio, coarsePointer ? 1.15 : 1.65))
  renderer.setSize(innerWidth, innerHeight)
})

document.addEventListener('contextmenu', event => event.preventDefault())
updateHud()
requestAnimationFrame(loop)
