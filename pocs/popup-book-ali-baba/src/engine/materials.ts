import {
  AddEquation,
  type Color,
  CustomBlending,
  DoubleSide,
  ShaderMaterial,
  SrcColorFactor,
  type Texture,
  Vector3,
  ZeroFactor,
} from "three";

export const MASK_LAYER = 1;

export const lampUniform = { value: new Vector3(0, 2.2, -4.2) };

const maskVertex = `
uniform vec3 lamp;
uniform float soft;
varying vec2 vUv;
varying float vBlur;
void main() {
  vUv = uv;
  vec4 local = vec4(position, 1.0);
  #ifdef USE_INSTANCING
  local = instanceMatrix * local;
  #endif
  vec4 world = modelMatrix * local;
  float toScreen = max(-world.z, 0.0);
  float toLamp = max(world.z - lamp.z, 0.08);
  vBlur = soft * clamp(log2(1.0 + 26.0 * toScreen / toLamp), 0.0, 5.5);
  gl_Position = projectionMatrix * viewMatrix * world;
}`;

const maskFragment = `
uniform sampler2D map;
uniform float hasMap;
uniform float density;
varying vec2 vUv;
varying float vBlur;
void main() {
  float a = 1.0;
  if (hasMap > 0.5) {
    float raw = texture2D(map, vUv, vBlur).a;
    float w = 0.05 + vBlur * 0.09;
    a = smoothstep(0.5 - w, 0.5 + w, raw);
  }
  if (a < 0.002) discard;
  gl_FragColor = vec4(vec3(1.0 - a * density), 1.0);
}`;

export function maskMaterial(map: Texture | null, density = 1, soft = 1): ShaderMaterial {
  return new ShaderMaterial({
    uniforms: {
      map: { value: map },
      hasMap: { value: map ? 1 : 0 },
      density: { value: density },
      soft: { value: soft },
      lamp: lampUniform,
    },
    vertexShader: maskVertex,
    fragmentShader: maskFragment,
    side: DoubleSide,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: CustomBlending,
    blendEquation: AddEquation,
    blendSrc: ZeroFactor,
    blendDst: SrcColorFactor,
  });
}

const screenVertex = `
varying vec2 vUv;
varying vec3 vWorld;
void main() {
  vUv = uv;
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}`;

const screenFragment = `
uniform sampler2D mask;
uniform sampler2D paper;
uniform vec3 lamp;
uniform vec3 top;
uniform vec3 bottom;
uniform float level;
varying vec2 vUv;
varying vec3 vWorld;
void main() {
  vec3 m = texture2D(mask, vec2(1.0 - vUv.x, vUv.y)).rgb;
  m = mix(vec3(0.012), vec3(1.0), m);
  vec3 toLamp = lamp - vWorld;
  float d2 = dot(toLamp, toLamp);
  float cosA = abs(toLamp.z) / sqrt(d2);
  float raw = cosA * pow(18.6 / (1.0 + d2), 0.75);
  float intensity = level * 1.6 * (1.0 - exp(-raw * 0.95));
  vec3 tint = mix(bottom, top, smoothstep(0.0, 1.0, vUv.y));
  float grain = texture2D(paper, vUv * vec2(2.0, 1.2)).r;
  float edge = smoothstep(0.0, 0.035, vUv.x) * smoothstep(1.0, 0.965, vUv.x) * smoothstep(0.0, 0.05, vUv.y) * smoothstep(1.0, 0.95, vUv.y);
  vec3 lit = tint * intensity * m * (0.8 + 0.35 * grain) * (0.55 + 0.45 * edge);
  gl_FragColor = vec4(lit * 1.4 + tint * 0.006 * level, 1.0);
}`;

export function screenMaterial(mask: Texture, paper: Texture, top: Color, bottom: Color): ShaderMaterial {
  return new ShaderMaterial({
    uniforms: {
      mask: { value: mask },
      paper: { value: paper },
      lamp: lampUniform,
      top: { value: top },
      bottom: { value: bottom },
      level: { value: 1 },
    },
    vertexShader: screenVertex,
    fragmentShader: screenFragment,
    side: DoubleSide,
  });
}
