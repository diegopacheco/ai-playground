export const VERTEX = `
attribute vec2 aPos;
void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }
`;

export const FRAGMENT = `
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform vec2 uMouse;
uniform float uScroll;
uniform float uShock;
uniform float uPledge;
uniform float uShift;

#define PI 3.14159265

float gT, gMelt, gMorph, gSnap, gPush;
float gEye, gRing, gTok;
mat2 gRy, gRx;

float hash1(float n){ return fract(sin(n) * 43758.5453); }
float hash2(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
mat2 rot(float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float smin(float a, float b, float k){ float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }
float smax(float a, float b, float k){ return -smin(-a, -b, k); }
float sdBox(vec3 p, vec3 b){ vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float sdRBox(vec3 p, vec3 b, float r){ return sdBox(p, b - r) - r; }
float sdCap(vec3 p, vec3 a, vec3 b, float r){ vec3 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h) - r; }
float sdTorus(vec3 p, vec2 t){ vec2 q = vec2(length(p.xy) - t.x, p.z); return length(q) - t.y; }

float robot(vec3 p){
  float n = gMelt;
  vec3 q = p;
  q.y += n * 0.14 * sin(q.x * 3.0 + gT * 1.7) * sin(q.z * 2.0 + gT);
  float d = sdRBox(q, vec3(0.95, 0.82, 0.82), 0.28);
  vec3 e = vec3(abs(q.x) - 0.36, q.y - 0.12, q.z - 0.82);
  d = smax(d, -(length(e) - 0.21), 0.05);
  vec3 g = q; g.x = mod(g.x + 0.1, 0.2) - 0.1;
  float slot = max(sdBox(vec3(g.x, q.y + 0.4, q.z - 0.82), vec3(0.035, 0.09, 0.12)), abs(q.x) - 0.5);
  d = max(d, -slot);
  d = min(d, sdCap(q, vec3(0.0, 0.8, 0.0), vec3(0.0, 1.25, 0.0), 0.045));
  d = min(d, sdRBox(vec3(abs(q.x) - 1.0, q.y, q.z), vec3(0.12, 0.3, 0.3), 0.08));
  gEye = min(length(e + vec3(0.0, 0.0, 0.07)) - 0.13, length(q - vec3(0.0, 1.32, 0.0)) - 0.11);
  vec2 id = floor((q.xz + 0.18) / 0.36);
  vec3 r = q; r.xz = mod(r.xz + 0.18, 0.36) - 0.18;
  float h = hash2(id);
  float inFoot = step(max(abs(id.x), abs(id.y)), 2.0);
  float len = n * (0.25 + 1.1 * h) * (0.75 + 0.25 * sin(gT * (0.6 + h) + h * 6.28));
  float drip = sdCap(r, vec3(0.0, -0.6, 0.0), vec3(0.0, -0.75 - len, 0.0), (0.05 + 0.06 * h) * n + 0.001);
  float fall = mod(gT * (0.5 + h * 0.6) + h * 9.0, 1.0);
  float drop = length(r - vec3(0.0, -0.9 - len - fall * 2.2, 0.0)) - 0.07 * n * (1.0 - fall * 0.5);
  d = smin(d, mix(10.0, drip, inFoot), 0.25);
  d = min(d, mix(10.0, drop, inFoot * step(0.35, h)));
  d += n * 0.035 * sin(q.x * 9.0 + gT * 2.0) * sin(q.y * 8.0) * sin(q.z * 7.0 - gT);
  return d;
}

float gyroid(vec3 p){
  vec3 s = p * 9.0 + 0.9 * sin(p.yzx * 5.0 + gT * 0.2);
  return sin(s.x) * cos(s.y) + sin(s.y) * cos(s.z) + sin(s.z) * cos(s.x);
}

float folds(vec3 p){
  return 1.0 - smoothstep(0.0, 0.4, abs(gyroid(p)));
}

float brain(vec3 p){
  vec3 h = vec3(p.x, p.y, abs(p.z) - 0.46);
  float d = (length(h * vec3(0.85, 1.25, 1.55)) - 0.9) * 0.6;
  d = smin(d, (length((p - vec3(0.62, -0.5, 0.0)) * vec3(1.6, 2.2, 1.1)) - 0.55) * 0.45, 0.12);
  d += 0.045 * folds(p);
  return smin(d, sdCap(p, vec3(0.2, -0.4, 0.0), vec3(0.4, -1.25, 0.0), 0.14), 0.12);
}

vec3 toObj(vec3 p){
  p.y -= 0.15 + 0.08 * sin(gT * 1.1);
  p.xz = gRy * p.xz;
  p.yz = gRx * p.yz;
  return p;
}

float object(vec3 p){
  vec3 o = toObj(p);
  gEye = 10.0;
  float d;
  if (gMorph < 0.001) d = robot(o);
  else if (gMorph > 0.999) d = brain(o);
  else d = mix(robot(o), brain(o), gMorph);
  return d + sin(gMorph * PI) * 0.2 * sin(o.x * 7.0 + gT * 3.0) * sin(o.y * 6.0) * sin(o.z * 8.0 + gT * 2.0);
}

float ring(vec3 p){
  if (gPush > 0.995) return 10.0;
  vec3 q = p - vec3(0.0, 0.15, gPush * 7.0);
  q.xz *= rot((1.0 - gSnap) * (gT * 2.2 + 1.2));
  q.xy *= rot((1.0 - gSnap) * 0.6);
  float rr = 1.7 + gPush * 0.7;
  float t = sdTorus(q, vec2(rr, 0.11));
  vec3 a = vec3(-0.707, 0.707, 0.0) * rr * gSnap;
  return min(t, sdCap(q, a, -a, 0.11));
}

float tokens(vec3 p){
  vec3 q = p; q.y -= 0.1;
  q.xz *= rot(gT * 0.25);
  float sec = 2.0 * PI / 16.0;
  float a = atan(q.z, q.x);
  float id = floor(a / sec + 0.5);
  a -= id * sec;
  float rl = length(q.xz);
  vec3 c = vec3(rl * cos(a) - 2.7, q.y, rl * sin(a));
  float h = hash1(id + 3.0);
  float chaos = 1.0 - gMorph;
  c.y -= chaos * sin(gT * (0.7 + h) + h * 20.0) * 1.1;
  c.x -= chaos * (h - 0.5) * 0.8;
  c.xy *= rot(gT * (1.0 + h) * chaos + 0.785 * gMorph);
  c.yz *= rot((gT * 0.7 + h * 6.0) * chaos);
  return sdRBox(c, vec3(0.11), 0.03);
}

float ground(vec3 p){
  float r = length(p.xz);
  return p.y + 1.95 - (1.0 - gMorph) * 0.05 * sin(r * 7.0 - gT * 2.5) * exp(-r * 0.35);
}

vec2 map(vec3 p){
  float o = object(p);
  gRing = ring(p);
  gTok = tokens(p);
  vec2 res = vec2(o, 1.0);
  if (gRing < res.x) res = vec2(gRing, 3.0);
  if (gTok < res.x) res = vec2(gTok, 5.0);
  float g = ground(p);
  if (g < res.x) res = vec2(g, 4.0);
  if (gEye < res.x) res = vec2(gEye, 2.0);
  return res;
}

vec3 normal(vec3 p){
  vec2 e = vec2(0.0015, 0.0);
  return normalize(vec3(
    map(p + e.xyy).x - map(p - e.xyy).x,
    map(p + e.yxy).x - map(p - e.yxy).x,
    map(p + e.yyx).x - map(p - e.yyx).x));
}

float occlusion(vec3 p, vec3 n){
  float o = 0.0, s = 1.0;
  for (int i = 1; i <= 4; i++){
    float h = 0.07 * float(i);
    o += (h - map(p + n * h).x) * s;
    s *= 0.6;
  }
  return clamp(1.0 - 2.5 * o, 0.0, 1.0);
}

float shadow(vec3 ro, vec3 rd){
  float res = 1.0, t = 0.04;
  for (int i = 0; i < 22; i++){
    float h = map(ro + rd * t).x;
    res = min(res, 10.0 * h / t);
    t += clamp(h, 0.04, 0.35);
    if (res < 0.01 || t > 6.0) break;
  }
  return clamp(res, 0.0, 1.0);
}

vec3 sky(vec3 rd){
  vec3 c = mix(vec3(0.05, 0.015, 0.03), vec3(0.006, 0.004, 0.014), clamp(rd.y * 1.6 + 0.3, 0.0, 1.0));
  c += vec3(0.95, 0.32, 0.05) * 0.22 * exp(-abs(rd.y + 0.02) * 8.0);
  vec3 md = normalize(vec3(0.42, 0.36, -1.0));
  float m = max(dot(rd, md), 0.0);
  float disk = smoothstep(0.9975, 0.9985, m);
  float crater = 0.8 + 0.2 * sin(rd.x * 140.0) * sin(rd.y * 120.0 + 1.3);
  c += vec3(1.0, 0.55, 0.2) * disk * crater * 1.6;
  c += vec3(1.0, 0.4, 0.1) * (pow(m, 300.0) * 0.8 + pow(m, 14.0) * 0.1);
  return c;
}

float march(vec3 ro, vec3 rd, int steps, out float mat, out vec3 glow){
  float t = 0.0;
  mat = 0.0;
  glow = vec3(0.0);
  for (int i = 0; i < 96; i++){
    if (i >= steps) break;
    vec3 p = ro + rd * t;
    vec2 h = map(p);
    float chaos = 1.0 - gMorph;
    glow += vec3(1.0, 0.06, 0.12) * exp(-gRing * 9.0) * 0.022;
    glow += vec3(0.55, 1.0, 0.15) * exp(-gEye * 14.0) * 0.03 * chaos;
    glow += mix(vec3(1.0, 0.72, 0.45), vec3(1.0, 0.35, 0.05), chaos) * exp(-gTok * 18.0) * 0.012;
    float w = uShock * 5.5;
    glow += vec3(1.0, 0.6, 0.3) * exp(-abs(length(p - vec3(0.0, 0.15, 0.0)) - w) * 14.0) * exp(-uShock * 1.4) * 0.05;
    if (h.x < 0.0012 * t + 0.0005){ mat = h.y; return t; }
    t += h.x * 0.75;
    if (t > 22.0) break;
  }
  mat = 0.0;
  return t;
}

vec3 surface(vec3 p, vec3 n, vec3 rd, float mat){
  vec3 lig = normalize(vec3(-0.5, 0.8, 0.6));
  vec3 back = normalize(vec3(0.6, 0.2, -0.8));
  float dif = max(dot(n, lig), 0.0) * shadow(p + n * 0.01, lig);
  float occ = occlusion(p, n);
  float fre = pow(1.0 - max(dot(n, -rd), 0.0), 4.0);
  vec3 ref = reflect(rd, n);
  float spe = pow(max(dot(ref, lig), 0.0), 40.0);
  float rim = pow(max(dot(n, back), 0.0), 2.0);
  vec3 env = sky(ref);
  vec3 col;
  if (mat < 1.5){
    vec3 o = toObj(p);
    float goo = gMelt * smoothstep(-0.35, -0.95, o.y);
    vec3 chrome = vec3(0.05, 0.045, 0.06) * (0.3 + dif) + env * (0.35 + 0.65 * fre) + vec3(1.0) * spe * 0.8;
    vec3 slop = vec3(1.0, 0.36, 0.03) * (0.25 + dif * 0.9) + vec3(1.0, 0.8, 0.5) * spe + vec3(1.0, 0.3, 0.0) * fre * 0.6;
    vec3 robo = mix(chrome, slop, goo);
    vec3 skin = mix(vec3(1.0, 0.58, 0.52), vec3(0.4, 0.07, 0.1), folds(o));
    vec3 brainCol = skin * (0.3 + dif * 0.9) + vec3(1.0, 0.4, 0.35) * fre * 0.8 + vec3(1.0, 0.9, 0.8) * spe * 0.35;
    brainCol += vec3(1.0, 0.25, 0.2) * pow(max(dot(-n, lig) * 0.5 + 0.5, 0.0), 3.0) * 0.18;
    col = mix(robo, brainCol, gMorph);
    col += vec3(1.0, 0.35, 0.05) * rim * 0.9;
  } else if (mat < 2.5){
    col = mix(vec3(0.55, 1.0, 0.15) * 3.0, vec3(1.0, 0.4, 0.3), gMorph) * (0.8 + 0.2 * sin(gT * 9.0));
  } else if (mat < 3.5){
    col = vec3(1.0, 0.06, 0.12) * (1.4 + dif) + env * fre * 0.5 + vec3(1.0) * spe;
  } else if (mat < 4.5){
    float grid = gMorph * (smoothstep(0.03, 0.0, abs(fract(p.x) - 0.5)) + smoothstep(0.03, 0.0, abs(fract(p.z) - 0.5)));
    col = mix(vec3(0.04, 0.012, 0.01), vec3(0.03, 0.02, 0.03), gMorph) * (0.4 + dif);
    col += vec3(1.0, 0.45, 0.1) * grid * 0.5 * exp(-length(p.xz) * 0.25);
    col += env * fre * 0.25;
  } else {
    col = mix(vec3(1.0, 0.4, 0.05), vec3(1.0, 0.85, 0.7), gMorph) * (0.6 + dif) + env * fre + vec3(1.0) * spe;
  }
  return col * mix(0.6, 1.0, occ);
}

void main(){
  vec2 frag = gl_FragCoord.xy;
  vec2 uv = (2.0 * frag - uRes) / uRes.y;
  gT = uTime;
  gMorph = clamp(max(smoothstep(0.52, 0.86, uScroll), uPledge), 0.0, 1.0);
  gMelt = (1.0 - gMorph) * (0.45 + 0.55 * smoothstep(0.0, 0.35, uScroll));
  gSnap = smoothstep(0.8, 3.0, uTime);
  gPush = max(smoothstep(0.24, 0.5, uScroll), uPledge);
  gRy = rot(-uMouse.x * 0.7 + (1.0 - gSnap) * 0.9);
  gRx = rot(uMouse.y * 0.35);

  float chaos = 1.0 - gMorph;
  float glitchOn = chaos * step(fract(uTime * 0.31), 0.06) + step(uShock, 0.25);
  float band = floor(uv.y * 22.0) + floor(uTime * 14.0);
  uv.x += glitchOn * (hash1(band) - 0.5) * 0.12 * step(0.7, hash1(band + 7.0));

  float yaw = uMouse.x * 0.25 + smoothstep(0.55, 1.0, uScroll) * 0.55;
  vec3 ro = vec3(5.4 * sin(yaw), 0.35 + uMouse.y * 0.4, 5.4 * cos(yaw));
  vec3 ta = vec3(0.0, 0.05, 0.0);
  vec3 ww = normalize(ta - ro);
  vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
  vec3 vv = cross(uu, ww);
  vec3 rd = normalize((uv.x + uShift) * uu + uv.y * vv + 1.9 * ww);

  float mat;
  vec3 glow;
  float t = march(ro, rd, 96, mat, glow);
  vec3 col = sky(rd);
  if (mat > 0.5){
    vec3 p = ro + rd * t;
    vec3 n = normal(p);
    col = surface(p, n, rd, mat);
    if (mat > 3.5 && mat < 4.5){
      vec3 rr = reflect(rd, n);
      float rm;
      vec3 rg;
      float rt = march(p + n * 0.02, rr, 48, rm, rg);
      vec3 rc = sky(rr);
      if (rm > 0.5 && rm < 3.5){
        vec3 rp = p + n * 0.02 + rr * rt;
        rc = surface(rp, normal(rp), rr, rm);
      }
      float f = mix(0.35, 0.18, gMorph);
      col += (rc + rg) * f;
    }
    col = mix(col, sky(rd), 1.0 - exp(-0.0025 * t * t));
  }
  col += glow;

  vec2 sp = frag / uRes.y * 38.0;
  sp.y -= uTime * 1.6;
  vec2 cell = floor(sp);
  float eh = hash2(cell);
  vec2 f = fract(sp) - 0.5 - (vec2(hash2(cell + 3.1), hash2(cell + 7.7)) - 0.5) * 0.7;
  float ember = step(0.93, eh) * smoothstep(0.08, 0.0, length(f)) * (0.5 + 0.5 * sin(uTime * 3.0 + eh * 40.0));
  col += mix(vec3(1.0, 0.35, 0.05), vec3(1.0, 0.85, 0.6), gMorph) * ember * 0.6;

  col = 1.0 - exp(-col * 1.25);
  col = pow(col, vec3(0.4545));
  vec2 q = frag / uRes;
  col *= 0.35 + 0.65 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.2);
  col *= 1.0 - 0.07 * chaos * (0.5 + 0.5 * sin(frag.y * 2.2));
  col += (hash2(frag + fract(uTime)) - 0.5) * 0.04;
  gl_FragColor = vec4(col, 1.0);
}
`;
