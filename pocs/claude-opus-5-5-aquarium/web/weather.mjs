import * as THREE from 'three';

const NOISE = `
varying vec3 vWPos;
varying vec3 vWNor;
uniform float uRust;
uniform float uAlgae;
uniform float uPlanks;
uniform float uBury;
float wHash(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float wNoise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(wHash(i), wHash(i + vec3(1, 0, 0)), f.x), mix(wHash(i + vec3(0, 1, 0)), wHash(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(wHash(i + vec3(0, 0, 1)), wHash(i + vec3(1, 0, 1)), f.x), mix(wHash(i + vec3(0, 1, 1)), wHash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float wFbm(vec3 p) {
  return 0.5 * wNoise(p) + 0.25 * wNoise(p * 2.03) + 0.125 * wNoise(p * 4.01) + 0.125 * wNoise(p * 8.07);
}
`;

const COLOR = `
float wBig = wFbm(vWPos * 55.0);
float wFine = wFbm(vWPos * 210.0 + 7.0);
float wRust = smoothstep(0.62 - uRust * 0.28, 0.72 - uRust * 0.28, wBig + wFine * 0.3) * uRust;
vec3 wRustC = mix(vec3(0.24, 0.1, 0.04), vec3(0.52, 0.24, 0.08), wFine);
diffuseColor.rgb = mix(diffuseColor.rgb, wRustC, wRust);
float wStreak = wNoise(vec3(vWPos.x * 260.0, vWPos.y * 18.0, vWPos.z * 260.0));
diffuseColor.rgb *= 0.78 + 0.3 * wFine - 0.12 * wStreak * uRust;
if (uPlanks > 0.0) {
  float wLine = abs(fract(vWPos.y * uPlanks) - 0.5);
  diffuseColor.rgb *= 0.72 + 0.28 * smoothstep(0.44, 0.4, wLine);
}
float wUp = clamp(normalize(vWNor).y, 0.0, 1.0);
float wAlgae = smoothstep(0.45, 0.8, wUp * 0.85 + wBig * 0.55 - 0.15) * uAlgae;
diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.16, 0.26, 0.1) * (0.7 + 0.7 * wFine), wAlgae);
float wSilt = smoothstep(uBury + 0.008, uBury - 0.002, vWPos.y);
diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.62, 0.53, 0.38), wSilt * 0.8);
`;

export function weathered(color, { rust = 0.5, algae = 0.6, planks = 0, roughness = 0.8, metalness = 0, bury = 0, extra = {} } = {}) {
  const m = new THREE.MeshStandardMaterial({ color, roughness, metalness, ...extra });
  const uniforms = { uRust: { value: rust }, uAlgae: { value: algae }, uPlanks: { value: planks }, uBury: { value: bury } };
  m.userData.weather = uniforms;
  m.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;\nvarying vec3 vWNor;')
      .replace('#include <project_vertex>', '#include <project_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvWNor = mat3(modelMatrix) * objectNormal;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${NOISE}`)
      .replace('#include <color_fragment>', `#include <color_fragment>\n${COLOR}`)
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.95, max(wRust, wAlgae));')
      .replace('#include <metalnessmap_fragment>', '#include <metalnessmap_fragment>\nmetalnessFactor *= 1.0 - max(wRust, wAlgae);');
  };
  m.customProgramCacheKey = () => 'weathered';
  return m;
}
