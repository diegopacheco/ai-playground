import { VERTEX, FRAGMENT } from "./shader.mjs";

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function program(gl) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(prog));
  }
  return prog;
}

export function startScene(canvas, state) {
  const gl = canvas.getContext("webgl", { antialias: false, powerPreference: "high-performance" });
  if (!gl) throw new Error("webgl unavailable");
  const prog = program(gl);
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const pos = gl.getAttribLocation(prog, "aPos");
  gl.enableVertexAttribArray(pos);
  gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);

  const u = {};
  for (const name of ["uRes", "uTime", "uMouse", "uScroll", "uShock", "uPledge", "uShift"]) {
    u[name] = gl.getUniformLocation(prog, name);
  }

  let scale = Math.min(window.devicePixelRatio || 1, 2) * 0.55;
  let frames = 0;
  let spent = 0;
  let last = performance.now();
  const mouse = { x: 0, y: 0 };
  const smooth = { x: 0, y: 0, scroll: 0 };

  function resize() {
    canvas.width = Math.max(1, Math.floor(window.innerWidth * scale));
    canvas.height = Math.max(1, Math.floor(window.innerHeight * scale));
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  function adapt(dt) {
    frames += 1;
    spent += dt;
    if (frames < 40) return;
    const avg = spent / frames;
    frames = 0;
    spent = 0;
    const next = avg > 26 ? scale * 0.82 : avg < 13 ? scale * 1.12 : scale;
    const clamped = Math.min(Math.max(next, 0.3), Math.min(window.devicePixelRatio || 1, 2));
    if (Math.abs(clamped - scale) > 0.01) {
      scale = clamped;
      resize();
    }
  }

  function frame(now) {
    const dt = now - last;
    last = now;
    adapt(dt);
    smooth.x += (mouse.x - smooth.x) * 0.06;
    smooth.y += (mouse.y - smooth.y) * 0.06;
    smooth.scroll += (state.scroll - smooth.scroll) * 0.08;
    const wide = window.innerWidth / window.innerHeight > 1.15;
    gl.uniform2f(u.uRes, canvas.width, canvas.height);
    gl.uniform1f(u.uTime, now / 1000);
    gl.uniform2f(u.uMouse, smooth.x, smooth.y);
    gl.uniform1f(u.uScroll, smooth.scroll);
    gl.uniform1f(u.uShock, (now - state.shockAt) / 1000);
    gl.uniform1f(u.uPledge, state.pledge);
    gl.uniform1f(u.uShift, wide ? -0.62 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    requestAnimationFrame(frame);
  }

  window.addEventListener("resize", resize);
  window.addEventListener("pointermove", (e) => {
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -((e.clientY / window.innerHeight) * 2 - 1);
  });
  resize();
  requestAnimationFrame(frame);
}
