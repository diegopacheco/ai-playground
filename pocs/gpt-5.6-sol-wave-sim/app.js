const canvas = document.getElementById('ocean')
const gl = canvas.getContext('webgl', { antialias: true, alpha: false })

if (!gl) {
  document.getElementById('unsupported').hidden = false
  throw new Error('WebGL is required')
}

const oceanVertex = `
attribute vec2 position;
varying vec2 uv;
void main() {
  uv = position;
  gl_Position = vec4(position, 0.0, 1.0);
}`

const oceanFragment = `
precision highp float;
varying vec2 uv;
uniform vec2 resolution;
uniform float time;
uniform float amplitude;
uniform float wavelength;
uniform float wind;
uniform float chop;
uniform float storm;
uniform vec3 camera;
uniform vec3 target;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1,0)), f.x), mix(hash(i + vec2(0,1)), hash(i + vec2(1)), f.x), f.y);
}

float wave(vec2 p) {
  float scale = 6.0 / wavelength;
  float speed = .32 + wind * .018;
  float t = time * speed;
  float h = sin(dot(p, normalize(vec2(.9,.42))) * scale + t * 1.7);
  h += .52 * sin(dot(p, normalize(vec2(-.38,.92))) * scale * 1.63 + t * 2.13 + 1.4);
  h += .24 * sin(dot(p, normalize(vec2(.68,-.73))) * scale * 2.87 + t * 2.9);
  h += chop * .18 * sin(dot(p, vec2(2.7,1.4)) * scale + t * 4.0);
  return h * amplitude * .32;
}

vec3 sky(vec3 rd) {
  float horizon = pow(1.0 - max(rd.y, 0.0), 4.0);
  vec3 zenith = mix(vec3(.11,.32,.41), vec3(.045,.12,.16), storm);
  vec3 low = mix(vec3(.72,.81,.79), vec3(.25,.34,.36), storm);
  vec3 col = mix(low, zenith, smoothstep(-.08,.72,rd.y));
  vec3 sunDir = normalize(vec3(-.45,.48,-.58));
  float sun = pow(max(dot(rd,sunDir),0.0), 620.0);
  float glow = pow(max(dot(rd,sunDir),0.0), 12.0);
  col += mix(vec3(1.0,.77,.42), vec3(.65,.77,.74), storm) * (sun * 8.0 + glow * .28);
  vec2 cloudP = rd.xz / max(rd.y + .24, .08) * .75 + vec2(time * .012,0.0);
  float clouds = smoothstep(.46,.76,noise(cloudP) * .65 + noise(cloudP * 2.2) * .35);
  col = mix(col, mix(vec3(.8,.84,.79),vec3(.25,.3,.31),storm), clouds * smoothstep(.02,.32,rd.y) * .45);
  col += horizon * mix(vec3(.18,.25,.23),vec3(.05,.07,.07),storm);
  return col;
}

void main() {
  vec2 p = uv;
  p.x *= resolution.x / resolution.y;
  vec3 forward = normalize(target - camera);
  vec3 right = normalize(cross(forward, vec3(0,1,0)));
  vec3 up = cross(right, forward);
  vec3 rd = normalize(forward + p.x * right * .68 + p.y * up * .68);
  vec3 ro = camera;
  vec3 col = sky(rd);
  if (rd.y < .12) {
    float dist = max(.1, -ro.y / min(rd.y, -.001));
    for (int i = 0; i < 7; i++) {
      vec3 pos = ro + rd * dist;
      float h = wave(pos.xz);
      dist += (h - pos.y) / rd.y * .72;
    }
    vec3 pos = ro + rd * dist;
    if (dist > 0.0 && dist < 160.0) {
      float e = .035;
      float h = wave(pos.xz);
      vec3 n = normalize(vec3(wave(pos.xz-vec2(e,0.0))-wave(pos.xz+vec2(e,0.0)), 2.0*e, wave(pos.xz-vec2(0.0,e))-wave(pos.xz+vec2(0.0,e))));
      vec3 refl = reflect(rd,n);
      float fresnel = .025 + .975 * pow(1.0 - max(dot(-rd,n),0.0),5.0);
      vec3 deep = mix(vec3(.006,.12,.16),vec3(.01,.055,.066),storm);
      vec3 reflected = sky(refl);
      float diffuse = max(dot(n,normalize(vec3(-.45,.48,-.58))),0.0);
      col = mix(deep + diffuse * vec3(.025,.13,.14), reflected, fresnel * .9 + .1);
      float foam = smoothstep(.72,.98,abs(h) / max(amplitude*.32,.01) + chop * .13) * chop;
      col = mix(col,vec3(.72,.83,.8),foam*.48);
      float fog = 1.0-exp(-dist*.012);
      col = mix(col,sky(normalize(vec3(rd.x,.02,rd.z))),fog);
    }
  }
  float vignette = 1.0 - dot(uv*.48,uv*.48);
  col *= .84 + .16*vignette;
  col = col / (col + vec3(.78));
  col = pow(col,vec3(.88));
  gl_FragColor = vec4(col,1.0);
}`

const meshVertex = `
precision highp float;
attribute vec3 position;
attribute vec3 normal;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
varying vec3 worldPosition;
varying vec3 worldNormal;
void main() {
  vec4 world = model * vec4(position,1.0);
  worldPosition = world.xyz;
  worldNormal = normalize(mat3(model) * normal);
  gl_Position = projection * view * world;
}`

const meshFragment = `
precision highp float;
varying vec3 worldPosition;
varying vec3 worldNormal;
uniform vec3 color;
uniform float gloss;
uniform float time;
uniform float amplitude;
uniform float wavelength;
uniform float wind;
uniform float chop;
float wave(vec2 p) {
  float scale = 6.0 / wavelength;
  float t = time * (.32 + wind * .018);
  float h = sin(dot(p,normalize(vec2(.9,.42)))*scale+t*1.7);
  h += .52*sin(dot(p,normalize(vec2(-.38,.92)))*scale*1.63+t*2.13+1.4);
  h += .24*sin(dot(p,normalize(vec2(.68,-.73)))*scale*2.87+t*2.9);
  h += chop*.18*sin(dot(p,vec2(2.7,1.4))*scale+t*4.0);
  return h*amplitude*.32;
}
void main() {
  if (worldPosition.y < wave(worldPosition.xz) - .025) discard;
  vec3 n = normalize(worldNormal);
  vec3 light = normalize(vec3(-.45,.65,.58));
  vec3 viewDir = normalize(vec3(0,3.0,7.0)-worldPosition);
  float diffuse = max(dot(n,light),0.0);
  float rim = pow(1.0-max(dot(n,viewDir),0.0),3.0);
  float spec = pow(max(dot(reflect(-light,n),viewDir),0.0),mix(24.0,110.0,gloss));
  vec3 shaded = color*(.34+diffuse*.72)+spec*vec3(1.0,.9,.72)*(.3+gloss)+rim*color*.18;
  gl_FragColor = vec4(shaded,1.0);
}`

function shader(type, source) {
  const value = gl.createShader(type)
  gl.shaderSource(value, source)
  gl.compileShader(value)
  if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(value))
  return value
}

function program(vertex, fragment) {
  const value = gl.createProgram()
  gl.attachShader(value, shader(gl.VERTEX_SHADER, vertex))
  gl.attachShader(value, shader(gl.FRAGMENT_SHADER, fragment))
  gl.linkProgram(value)
  if (!gl.getProgramParameter(value, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(value))
  return value
}

const oceanProgram = program(oceanVertex, oceanFragment)
const duckProgram = program(meshVertex, meshFragment)

const quadBuffer = gl.createBuffer()
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer)
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW)
const quadPosition = gl.getAttribLocation(oceanProgram, 'position')

function sphere(rows = 20, columns = 28) {
  const vertices = []
  const indices = []
  for (let y = 0; y <= rows; y++) {
    const v = y / rows
    const phi = v * Math.PI
    for (let x = 0; x <= columns; x++) {
      const u = x / columns
      const theta = u * Math.PI * 2
      const px = Math.sin(phi) * Math.cos(theta)
      const py = Math.cos(phi)
      const pz = Math.sin(phi) * Math.sin(theta)
      vertices.push(px,py,pz,px,py,pz)
    }
  }
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      const a = y * (columns + 1) + x
      const b = a + columns + 1
      indices.push(a,b,a+1,b,a+1,b+1)
    }
  }
  return { vertices: new Float32Array(vertices), indices: new Uint16Array(indices) }
}

const sphereData = sphere()
const sphereBuffer = gl.createBuffer()
gl.bindBuffer(gl.ARRAY_BUFFER, sphereBuffer)
gl.bufferData(gl.ARRAY_BUFFER, sphereData.vertices, gl.STATIC_DRAW)
const meshPosition = gl.getAttribLocation(duckProgram,'position')
const meshNormal = gl.getAttribLocation(duckProgram,'normal')
const sphereIndices = gl.createBuffer()
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,sphereIndices)
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,sphereData.indices,gl.STATIC_DRAW)

const controls = {
  height: document.getElementById('height'),
  length: document.getElementById('length'),
  speed: document.getElementById('speed'),
  chop: document.getElementById('chop'),
  storm: document.getElementById('storm')
}

const outputs = {
  height: document.getElementById('heightValue'),
  length: document.getElementById('lengthValue'),
  speed: document.getElementById('speedValue'),
  chop: document.getElementById('chopValue')
}

const presets = {
  calm: [0.35,8.5,5,18,false],
  swell: [1.15,5.8,14,46,false],
  squall: [2.05,3.1,28,84,true]
}

function updateControls() {
  outputs.height.value = `${Number(controls.height.value).toFixed(2)} m`
  outputs.length.value = `${Number(controls.length.value).toFixed(1)} m`
  outputs.speed.value = `${controls.speed.value} kn`
  outputs.chop.value = `${controls.chop.value}%`
  Object.values(controls).filter(control => control.type === 'range').forEach(control => {
    const fill = (control.value - control.min) / (control.max - control.min) * 100
    control.style.setProperty('--fill', `${fill}%`)
  })
}

Object.values(controls).forEach(control => control.addEventListener('input', () => {
  updateControls()
  document.querySelectorAll('[data-preset]').forEach(button => button.classList.remove('active'))
}))

document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
  const values = presets[button.dataset.preset]
  controls.height.value = values[0]
  controls.length.value = values[1]
  controls.speed.value = values[2]
  controls.chop.value = values[3]
  controls.storm.checked = values[4]
  document.querySelectorAll('[data-preset]').forEach(item => item.classList.toggle('active', item === button))
  updateControls()
}))

document.getElementById('collapse').addEventListener('click', event => {
  const panel = document.querySelector('.panel')
  panel.classList.toggle('collapsed')
  event.currentTarget.textContent = panel.classList.contains('collapsed') ? '+' : '−'
})

let running = true
document.getElementById('motionToggle').addEventListener('click', event => {
  running = !running
  event.currentTarget.setAttribute('aria-pressed', String(running))
  document.getElementById('motionLabel').textContent = running ? 'LIVE MOTION' : 'MOTION PAUSED'
})

let yaw = 0
let pitch = 0
let zoom = 7.2
let dragging = false
let previous = [0,0]
canvas.addEventListener('pointerdown', event => {
  dragging = true
  previous = [event.clientX,event.clientY]
  canvas.setPointerCapture(event.pointerId)
})
canvas.addEventListener('pointerup', () => dragging = false)
canvas.addEventListener('pointermove', event => {
  if (!dragging) return
  yaw += (event.clientX-previous[0])*.004
  pitch = Math.max(-.22,Math.min(.32,pitch+(event.clientY-previous[1])*.003))
  previous = [event.clientX,event.clientY]
})
canvas.addEventListener('wheel', event => {
  zoom = Math.max(4.8,Math.min(11,zoom+event.deltaY*.005))
}, { passive: true })

function identity() {
  return new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])
}

function multiply(a,b) {
  const out = new Float32Array(16)
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) out[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3]
  return out
}

function transform(position, rotation, scale) {
  const [x,y,z] = position
  const [rx,ry,rz] = rotation
  const cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry),cz=Math.cos(rz),sz=Math.sin(rz)
  const rotationMatrix = new Float32Array([
    cy*cz, sx*sy*cz+cx*sz, -cx*sy*cz+sx*sz, 0,
    -cy*sz, -sx*sy*sz+cx*cz, cx*sy*sz+sx*cz, 0,
    sy, -sx*cy, cx*cy, 0,
    0,0,0,1
  ])
  rotationMatrix[0]*=scale[0]; rotationMatrix[1]*=scale[0]; rotationMatrix[2]*=scale[0]
  rotationMatrix[4]*=scale[1]; rotationMatrix[5]*=scale[1]; rotationMatrix[6]*=scale[1]
  rotationMatrix[8]*=scale[2]; rotationMatrix[9]*=scale[2]; rotationMatrix[10]*=scale[2]
  rotationMatrix[12]=x; rotationMatrix[13]=y; rotationMatrix[14]=z
  return rotationMatrix
}

function perspective(fov,aspect,near,far) {
  const f=1/Math.tan(fov/2), nf=1/(near-far)
  return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0])
}

function lookAt(eye,center) {
  let zx=eye[0]-center[0],zy=eye[1]-center[1],zz=eye[2]-center[2]
  let length=Math.hypot(zx,zy,zz); zx/=length;zy/=length;zz/=length
  let xx=zz,xz=-zx; length=Math.hypot(xx,xz);xx/=length;xz/=length
  const yx=zy*xz,yz=-zy*xx,yy=zz*xx-zx*xz
  return new Float32Array([xx,yx,zx,0,0,yy,zy,0,xz,yz,zz,0,-xx*eye[0]-xz*eye[2],-yx*eye[0]-yy*eye[1]-yz*eye[2],-zx*eye[0]-zy*eye[1]-zz*eye[2],1])
}

function waveAt(x,z,t) {
  const amplitude=Number(controls.height.value), wavelength=Number(controls.length.value), wind=Number(controls.speed.value), chop=Number(controls.chop.value)/100
  const scale=6/wavelength, time=t*(.32+wind*.018)
  let h=Math.sin((x*.9+z*.42)/Math.hypot(.9,.42)*scale+time*1.7)
  h+=.52*Math.sin((x*-.38+z*.92)/Math.hypot(.38,.92)*scale*1.63+time*2.13+1.4)
  h+=.24*Math.sin((x*.68+z*-.73)/Math.hypot(.68,.73)*scale*2.87+time*2.9)
  h+=chop*.18*Math.sin((x*2.7+z*1.4)*scale+time*4)
  return h*amplitude*.32
}

const parts = [
  [[0,.46,0],[0,0,0],[.72,.56,.56],[1,.69,.035],.72],
  [[.34,.95,0],[0,0,-.04],[.43,.44,.42],[1,.73,.05],.8],
  [[.71,.92,0],[0,0,0],[.25,.13,.30],[1,.31,.015],.58],
  [[.22,.73,.48],[.12,0,-.34],[.48,.15,.27],[1,.62,.025],.67],
  [[.22,.73,-.48],[-.12,0,-.34],[.48,.15,.27],[1,.62,.025],.67],
  [[.56,1.09,.34],[0,0,0],[.075,.075,.055],[.018,.016,.012],.95],
  [[.56,1.09,-.34],[0,0,0],[.075,.075,.055],[.018,.016,.012],.95],
  [[-.64,.56,0],[0,0,.38],[.27,.2,.2],[1,.69,.035],.65]
]

function setUniform(programValue,name,method,...values) {
  const location=gl.getUniformLocation(programValue,name)
  gl[method](location,...values)
}

let simulatedTime=0
let lastTime=performance.now()
let frameCount=0
let fpsTime=lastTime

function resize() {
  const ratio=Math.min(devicePixelRatio,2)
  const width=Math.floor(canvas.clientWidth*ratio),height=Math.floor(canvas.clientHeight*ratio)
  if (canvas.width!==width||canvas.height!==height) { canvas.width=width;canvas.height=height }
  gl.viewport(0,0,width,height)
}

function render(now) {
  resize()
  const delta=Math.min((now-lastTime)/1000,.05)
  lastTime=now
  if (running) simulatedTime+=delta
  const camera=[Math.sin(yaw)*zoom,3.15+pitch*4,Math.cos(yaw)*zoom]
  const target=[0,.2,0]
  const amplitude=Number(controls.height.value), wavelength=Number(controls.length.value), wind=Number(controls.speed.value), chop=Number(controls.chop.value)/100, storm=controls.storm.checked?1:0
  gl.disable(gl.DEPTH_TEST)
  gl.useProgram(oceanProgram)
  gl.bindBuffer(gl.ARRAY_BUFFER,quadBuffer)
  gl.enableVertexAttribArray(quadPosition)
  gl.vertexAttribPointer(quadPosition,2,gl.FLOAT,false,0,0)
  setUniform(oceanProgram,'resolution','uniform2f',canvas.width,canvas.height)
  setUniform(oceanProgram,'time','uniform1f',simulatedTime)
  setUniform(oceanProgram,'amplitude','uniform1f',amplitude)
  setUniform(oceanProgram,'wavelength','uniform1f',wavelength)
  setUniform(oceanProgram,'wind','uniform1f',wind)
  setUniform(oceanProgram,'chop','uniform1f',chop)
  setUniform(oceanProgram,'storm','uniform1f',storm)
  setUniform(oceanProgram,'camera','uniform3fv',camera)
  setUniform(oceanProgram,'target','uniform3fv',target)
  gl.drawArrays(gl.TRIANGLES,0,6)

  gl.enable(gl.DEPTH_TEST)
  gl.clear(gl.DEPTH_BUFFER_BIT)
  gl.useProgram(duckProgram)
  gl.bindBuffer(gl.ARRAY_BUFFER,sphereBuffer)
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,sphereIndices)
  gl.enableVertexAttribArray(meshPosition)
  gl.vertexAttribPointer(meshPosition,3,gl.FLOAT,false,24,0)
  gl.enableVertexAttribArray(meshNormal)
  gl.vertexAttribPointer(meshNormal,3,gl.FLOAT,false,24,12)
  setUniform(duckProgram,'projection','uniformMatrix4fv',false,perspective(Math.PI/3,canvas.width/canvas.height,.1,100))
  setUniform(duckProgram,'view','uniformMatrix4fv',false,lookAt(camera,target))
  setUniform(duckProgram,'time','uniform1f',simulatedTime)
  setUniform(duckProgram,'amplitude','uniform1f',amplitude)
  setUniform(duckProgram,'wavelength','uniform1f',wavelength)
  setUniform(duckProgram,'wind','uniform1f',wind)
  setUniform(duckProgram,'chop','uniform1f',chop)
  const duckX=Math.sin(simulatedTime*.34)*1.35
  const duckZ=-.1+Math.sin(simulatedTime*.21)*.28
  const duckY=waveAt(duckX,duckZ,simulatedTime)-.18
  const slopeX=(waveAt(duckX+.08,duckZ,simulatedTime)-waveAt(duckX-.08,duckZ,simulatedTime))/.16
  const slopeZ=(waveAt(duckX,duckZ+.08,simulatedTime)-waveAt(duckX,duckZ-.08,simulatedTime))/.16
  const base=transform([duckX,duckY,duckZ],[-slopeZ*.32,Math.sin(simulatedTime*.18)*.12,slopeX*.3],[1,1,1])
  for (const part of parts) {
    const local=transform(part[0],part[1],part[2])
    setUniform(duckProgram,'model','uniformMatrix4fv',false,multiply(base,local))
    setUniform(duckProgram,'color','uniform3fv',part[3])
    setUniform(duckProgram,'gloss','uniform1f',part[4])
    gl.drawElements(gl.TRIANGLES,sphereData.indices.length,gl.UNSIGNED_SHORT,0)
  }
  frameCount++
  if (now-fpsTime>500) {
    document.getElementById('fps').textContent=Math.round(frameCount*1000/(now-fpsTime))
    frameCount=0
    fpsTime=now
  }
  requestAnimationFrame(render)
}

updateControls()
requestAnimationFrame(render)
