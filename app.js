import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Delaunator from 'delaunator';

// ---------------------------------------------------------------------------
// Constants & state
// ---------------------------------------------------------------------------

const PLANE_LONG = 10;      // world length of the plane's longer side
const PREVIEW_MAX = 512;    // longest side of the on-screen previews, in pixels
const TEX_MAX = 4096;       // largest exportable texture side
const MIN_VERTEX_GAP = 1e-3; // in world units; closer vertices are rejected
const STORAGE_KEY = 'normal-map-builder/v1';

const $ = (id) => document.getElementById(id);

function defaultState() {
  return {
    texW: 512,
    texH: 512,
    convention: 'opengl',
    previewKind: 'normal',
    dragMode: 'height',
    verts: [
      { u: 0, v: 0, h: 0, corner: true },
      { u: 1, v: 0, h: 0, corner: true },
      { u: 1, v: 1, h: 0, corner: true },
      { u: 0, v: 1, h: 0, corner: true },
      { u: 0.5, v: 0.5, h: 2 },
    ],
    light: { u: 0.2, v: 0.2, height: 6, type: 'point', ambient: 0.05 },
    gradient: {
      mode: 'smooth',
      stops: [{ p: 0, c: '#000000' }, { p: 1, c: '#ffffff' }],
      steps: ['#000000', '#555555', '#aaaaaa', '#ffffff'],
    },
  };
}

function isValidState(s) {
  return s && Array.isArray(s.verts) && s.verts.filter((v) => v.corner).length === 4
    && Number.isFinite(s.texW) && Number.isFinite(s.texH)
    && s.gradient && Array.isArray(s.gradient.stops) && s.gradient.stops.length >= 2
    && Array.isArray(s.gradient.steps) && s.gradient.steps.length >= 2 && s.light;
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    return isValidState(s) ? { ...defaultState(), ...s } : null;
  } catch { return null; }
}

let saveTimer = 0;
function saveState() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch { /* storage unavailable */ }
  }, 300);
}

let state = loadState() ?? defaultState();
let selected = -1;
let selectedStop = 0;

// ---------------------------------------------------------------------------
// Geometry: plane dimensions, triangulation, rasterization
// ---------------------------------------------------------------------------

function planeDims() {
  const a = state.texW / state.texH;
  return a >= 1 ? { w: PLANE_LONG, d: PLANE_LONG / a } : { w: PLANE_LONG * a, d: PLANE_LONG };
}

// u runs left→right (+X), v runs top→bottom of the texture (+Z, toward the default camera).
function toWorld(u, v) {
  const { w, d } = planeDims();
  return { x: (u - 0.5) * w, z: (v - 0.5) * d };
}
function toUV(x, z) {
  const { w, d } = planeDims();
  return { u: x / w + 0.5, v: z / d + 0.5 };
}

let mesh = { triangles: new Uint32Array(0), normals: new Float32Array(0) };

function triangulate() {
  const { w, d } = planeDims();
  const V = state.verts;
  const coords = new Float64Array(V.length * 2);
  V.forEach((p, i) => { coords[2 * i] = p.u * w; coords[2 * i + 1] = p.v * d; });
  const triangles = new Delaunator(coords).triangles;
  const n = triangles.length / 3;
  const normals = new Float32Array(n * 3);
  for (let t = 0; t < n; t++) {
    const a = V[triangles[3 * t]], b = V[triangles[3 * t + 1]], c = V[triangles[3 * t + 2]];
    const ax = a.u * w, az = a.v * d;
    const e1x = b.u * w - ax, e1y = b.h - a.h, e1z = b.v * d - az;
    const e2x = c.u * w - ax, e2y = c.h - a.h, e2z = c.v * d - az;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
    const len = Math.hypot(nx, ny, nz) || 1;
    normals[3 * t] = nx / len;
    normals[3 * t + 1] = ny / len;
    normals[3 * t + 2] = nz / len;
  }
  mesh = { triangles, normals };
}

/**
 * Rasterize the triangulated surface into a W×H grid sampled at pixel centers.
 * Returns the triangle index covering each pixel and (optionally) the
 * interpolated height there.
 */
function rasterize(W, H, withHeight) {
  const tri = new Int32Array(W * H).fill(-1);
  const height = withHeight ? new Float32Array(W * H) : null;
  const V = state.verts;
  const T = mesh.triangles;
  const EPS = 1e-7;
  for (let t = 0; t < T.length / 3; t++) {
    const A = V[T[3 * t]], B = V[T[3 * t + 1]], C = V[T[3 * t + 2]];
    const ax = A.u * W, ay = A.v * H, bx = B.u * W, by = B.v * H, cx = C.u * W, cy = C.v * H;
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (Math.abs(area) < 1e-12) continue;
    const inv = 1 / area;
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx) - 0.5));
    const x1 = Math.min(W - 1, Math.ceil(Math.max(ax, bx, cx) - 0.5));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy) - 0.5));
    const y1 = Math.min(H - 1, Math.ceil(Math.max(ay, by, cy) - 0.5));
    for (let py = y0; py <= y1; py++) {
      const y = py + 0.5;
      for (let px = x0; px <= x1; px++) {
        const k = py * W + px;
        if (tri[k] !== -1) continue;
        const x = px + 0.5;
        const w0 = ((bx - x) * (cy - y) - (by - y) * (cx - x)) * inv;
        if (w0 < -EPS) continue;
        const w1 = ((cx - x) * (ay - y) - (cy - y) * (ax - x)) * inv;
        if (w1 < -EPS) continue;
        const w2 = 1 - w0 - w1;
        if (w2 < -EPS) continue;
        tri[k] = t;
        if (height) height[k] = w0 * A.h + w1 * B.h + w2 * C.h;
      }
    }
  }
  return { W, H, tri, height };
}

function encodeNormalMap(r, data) {
  const N = mesh.normals;
  const flipG = state.convention === 'directx';
  for (let k = 0, n = r.W * r.H; k < n; k++) {
    const t = r.tri[k];
    let nx = 0, ny = 1, nz = 0;
    if (t >= 0) { nx = N[3 * t]; ny = N[3 * t + 1]; nz = N[3 * t + 2]; }
    // Tangent space: R = +X (right), G = up in the image (−Z for OpenGL), B = surface normal (+Y).
    const g = flipG ? nz : -nz;
    data[4 * k] = Math.round((nx * 0.5 + 0.5) * 255);
    data[4 * k + 1] = Math.round((g * 0.5 + 0.5) * 255);
    data[4 * k + 2] = Math.round((ny * 0.5 + 0.5) * 255);
    data[4 * k + 3] = 255;
  }
}

function heightRange() {
  let lo = Infinity, hi = -Infinity;
  for (const v of state.verts) { lo = Math.min(lo, v.h); hi = Math.max(hi, v.h); }
  return { lo, hi };
}

function encodeHeightMap(r, data) {
  const { lo, hi } = heightRange();
  const span = hi - lo;
  for (let k = 0, n = r.W * r.H; k < n; k++) {
    const g = span > 1e-9 ? Math.round(((r.height[k] - lo) / span) * 255) : 128;
    data[4 * k] = data[4 * k + 1] = data[4 * k + 2] = g;
    data[4 * k + 3] = 255;
  }
}

// ---------------------------------------------------------------------------
// Colors & gradient
// ---------------------------------------------------------------------------

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rgbToHex([r, g, b]) {
  return '#' + [r, g, b].map((x) => Math.round(x).toString(16).padStart(2, '0')).join('');
}

function sampleGradient(p) {
  const stops = [...state.gradient.stops].sort((a, b) => a.p - b.p);
  if (p <= stops[0].p) return hexToRgb(stops[0].c);
  const last = stops[stops.length - 1];
  if (p >= last.p) return hexToRgb(last.c);
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i], b = stops[i + 1];
    if (p >= a.p && p <= b.p) {
      const t = b.p > a.p ? (p - a.p) / (b.p - a.p) : 0;
      const ca = hexToRgb(a.c), cb = hexToRgb(b.c);
      return ca.map((x, j) => x + (cb[j] - x) * t);
    }
  }
  return hexToRgb(last.c);
}

/** 256-entry lookup table: light level → RGB. */
function buildLut() {
  const lut = new Uint8ClampedArray(256 * 3);
  const g = state.gradient;
  const stepRgb = g.steps.map(hexToRgb);
  for (let i = 0; i < 256; i++) {
    const c = g.mode === 'smooth'
      ? sampleGradient(i / 255)
      : stepRgb[Math.min(stepRgb.length - 1, Math.floor((i * stepRgb.length) / 256))];
    lut[3 * i] = c[0]; lut[3 * i + 1] = c[1]; lut[3 * i + 2] = c[2];
  }
  return lut;
}

// ---------------------------------------------------------------------------
// 2D previews
// ---------------------------------------------------------------------------

const normalCanvas = $('normal-canvas');
const litCanvas = $('lit-canvas');
const litFrame = $('lit-frame');
const lightMarker = $('light-marker');
let preview = null; // rasterization at preview resolution

function previewSize() {
  const s = Math.min(1, PREVIEW_MAX / Math.max(state.texW, state.texH));
  return { W: Math.max(1, Math.round(state.texW * s)), H: Math.max(1, Math.round(state.texH * s)) };
}

function sizeCanvas(c, W, H) {
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  // Show small textures as crisp texels instead of a blur.
  c.classList.toggle('pixelated', Math.max(W, H) < 256);
  fitCanvas(c);
}

const PREVIEW_MAX_CSS_HEIGHT = 340;
// Scale to the frame's width without exceeding the max height, keeping the aspect ratio.
function fitCanvas(c) {
  const frameW = c.parentElement.clientWidth - 2;
  if (frameW <= 0) return;
  const s = Math.min(frameW / c.width, PREVIEW_MAX_CSS_HEIGHT / c.height);
  c.style.width = `${c.width * s}px`;
  c.style.height = `${c.height * s}px`;
}

function updatePreviews() {
  const { W, H } = previewSize();
  preview = rasterize(W, H, true);
  sizeCanvas(normalCanvas, W, H);
  sizeCanvas(litCanvas, W, H);
  drawTexturePreview();
  drawLit();
}

function drawTexturePreview() {
  const { W, H } = preview;
  const ctx = normalCanvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  if (state.previewKind === 'height') encodeHeightMap(preview, img.data);
  else encodeNormalMap(preview, img.data);
  ctx.putImageData(img, 0, 0);
}

function drawLit() {
  const { W, H, tri, height } = preview;
  const N = mesh.normals;
  const { w, d } = planeDims();
  const L = state.light;
  const lx = (L.u - 0.5) * w, lz = (L.v - 0.5) * d, ly = L.height;
  const amb = L.ambient;
  const point = L.type === 'point';
  // A directional light shines from the plane's center toward the marker.
  let dx = lx, dy = ly, dz = lz;
  if (!point) { const len = Math.hypot(dx, dy, dz) || 1; dx /= len; dy /= len; dz /= len; }
  const lut = buildLut();
  const ctx = litCanvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  const data = img.data;
  for (let py = 0; py < H; py++) {
    const z = ((py + 0.5) / H - 0.5) * d;
    for (let px = 0; px < W; px++) {
      const k = py * W + px;
      const t = tri[k];
      let nx = 0, ny = 1, nz = 0;
      if (t >= 0) { nx = N[3 * t]; ny = N[3 * t + 1]; nz = N[3 * t + 2]; }
      let diff;
      if (point) {
        const x = ((px + 0.5) / W - 0.5) * w;
        const vx = lx - x, vy = ly - height[k], vz = lz - z;
        const len = Math.hypot(vx, vy, vz) || 1;
        diff = (nx * vx + ny * vy + nz * vz) / len;
      } else {
        diff = nx * dx + ny * dy + nz * dz;
      }
      const I = amb + (1 - amb) * Math.max(0, diff);
      const li = Math.min(255, Math.floor(I * 256)) * 3;
      data[4 * k] = lut[li]; data[4 * k + 1] = lut[li + 1]; data[4 * k + 2] = lut[li + 2];
      data[4 * k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  positionLightMarker();
}

function positionLightMarker() {
  const fr = litFrame.getBoundingClientRect();
  const cr = litCanvas.getBoundingClientRect();
  lightMarker.style.left = `${cr.left - fr.left + state.light.u * cr.width}px`;
  lightMarker.style.top = `${cr.top - fr.top + state.light.v * cr.height}px`;
}

// Light dragging
let lightDrag = false;
function moveLight(e) {
  const r = litCanvas.getBoundingClientRect();
  state.light.u = clamp((e.clientX - r.left) / r.width, 0, 1);
  state.light.v = clamp((e.clientY - r.top) / r.height, 0, 1);
  scheduleLit();
}
litFrame.addEventListener('pointerdown', (e) => {
  lightDrag = true;
  litFrame.setPointerCapture(e.pointerId);
  moveLight(e);
});
litFrame.addEventListener('pointermove', (e) => { if (lightDrag) moveLight(e); });
litFrame.addEventListener('pointerup', () => { lightDrag = false; saveState(); });
litFrame.addEventListener('pointercancel', () => { lightDrag = false; });

// ---------------------------------------------------------------------------
// Update scheduling (coalesce work into one pass per animation frame)
// ---------------------------------------------------------------------------

let pending = { geometry: false, lit: false, texture: false };
let frameQueued = false;
function queueFrame() {
  if (frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => {
    frameQueued = false;
    const p = pending;
    pending = { geometry: false, lit: false, texture: false };
    if (p.geometry) {
      triangulate();
      rebuildScene();
      updatePreviews();
      updateStats();
    } else {
      if (p.texture) drawTexturePreview();
      if (p.lit) drawLit();
    }
  });
}
function scheduleGeometry() { pending.geometry = true; queueFrame(); saveState(); }
function scheduleLit() { pending.lit = true; queueFrame(); }
function scheduleTexture() { pending.texture = true; queueFrame(); saveState(); }

// ---------------------------------------------------------------------------
// 3D editor
// ---------------------------------------------------------------------------

const viewport = $('viewport');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (err) {
  const el = $('load-error');
  el.hidden = false;
  el.textContent = 'WebGL is not available in this browser, so the 3D editor cannot start.';
  throw err;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
viewport.appendChild(renderer.domElement);
const canvasEl = renderer.domElement;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14171c);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);

// Our pointer handler is registered before OrbitControls so it can claim
// drags that start on a vertex handle.
canvasEl.addEventListener('pointerdown', onPointerDown);
canvasEl.addEventListener('pointermove', onPointerMove);
canvasEl.addEventListener('pointerup', onPointerUp);
canvasEl.addEventListener('pointercancel', onPointerUp);
canvasEl.addEventListener('contextmenu', (e) => e.preventDefault());

const controls = new OrbitControls(camera, canvasEl);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 2;
controls.maxDistance = 80;

// Distance at which the whole plane fits in view, given the viewport's aspect.
function fitDistance() {
  const { w, d } = planeDims();
  const halfV = THREE.MathUtils.degToRad(camera.fov) / 2;
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  const r = Math.hypot(w, d) / 2 * 1.08;
  return Math.max(r / Math.sin(halfV), r / Math.sin(halfH));
}
function setPerspectiveView() {
  const dist = fitDistance();
  camera.position.set(0, dist * Math.sin(0.85), dist * Math.cos(0.85));
  controls.target.set(0, 0, 0);
  controls.update();
}
function setTopView() {
  const { w, d } = planeDims();
  const halfV = THREE.MathUtils.degToRad(camera.fov) / 2;
  const dist = 1.12 * Math.max(d / 2 / Math.tan(halfV), w / 2 / (Math.tan(halfV) * camera.aspect));
  camera.position.set(0, dist, 0.0001);
  controls.target.set(0, 0, 0);
  controls.update();
}
setPerspectiveView();

scene.add(new THREE.HemisphereLight(0xdde6f5, 0x2a3240, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(6, 10, 4);
scene.add(sun);

const surfaceMat = new THREE.MeshStandardMaterial({
  color: 0x8193a8, roughness: 0.85, metalness: 0, flatShading: true, side: THREE.DoubleSide,
  polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
});
const surface = new THREE.Mesh(new THREE.BufferGeometry(), surfaceMat);
scene.add(surface);

const wire = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x0c0f13, transparent: true, opacity: 0.55 }),
);
scene.add(wire);

const outline = new THREE.LineLoop(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x5aa9ff, transparent: true, opacity: 0.5 }),
);
scene.add(outline);

const stems = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineDashedMaterial({ color: 0xffc857, dashSize: 0.12, gapSize: 0.1, transparent: true, opacity: 0.6 }),
);
scene.add(stems);

const handles = new THREE.Group();
scene.add(handles);
const handleGeo = new THREE.SphereGeometry(0.14, 16, 12);
const handleMats = {
  normal: new THREE.MeshBasicMaterial({ color: 0xffc857, depthTest: false }),
  corner: new THREE.MeshBasicMaterial({ color: 0x7dd3fc, depthTest: false }),
  selected: new THREE.MeshBasicMaterial({ color: 0xff4d6d, depthTest: false }),
};

function rebuildScene() {
  const V = state.verts;
  const T = mesh.triangles;
  const pos = new Float32Array(T.length * 3);
  const edges = new Float32Array(T.length * 6);
  for (let i = 0; i < T.length; i++) {
    const p = V[T[i]];
    const { x, z } = toWorld(p.u, p.v);
    pos.set([x, p.h, z], 3 * i);
  }
  for (let t = 0; t < T.length / 3; t++) {
    for (let e = 0; e < 3; e++) {
      const a = 3 * (3 * t + e), b = 3 * (3 * t + ((e + 1) % 3));
      edges.set([pos[a], pos[a + 1], pos[a + 2], pos[b], pos[b + 1], pos[b + 2]], 6 * (3 * t + e));
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.computeVertexNormals();
  surface.geometry.dispose();
  surface.geometry = g;

  const wg = new THREE.BufferGeometry();
  wg.setAttribute('position', new THREE.BufferAttribute(edges, 3));
  wire.geometry.dispose();
  wire.geometry = wg;

  const { w, d } = planeDims();
  const og = new THREE.BufferGeometry();
  og.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -w / 2, 0, -d / 2, w / 2, 0, -d / 2, w / 2, 0, d / 2, -w / 2, 0, d / 2,
  ]), 3));
  outline.geometry.dispose();
  outline.geometry = og;

  const stemPos = [];
  handles.clear();
  V.forEach((p, i) => {
    const { x, z } = toWorld(p.u, p.v);
    const m = new THREE.Mesh(
      handleGeo,
      i === selected ? handleMats.selected : p.corner ? handleMats.corner : handleMats.normal,
    );
    m.position.set(x, p.h, z);
    m.renderOrder = 10;
    m.userData.index = i;
    handles.add(m);
    if (Math.abs(p.h) > 1e-4) stemPos.push(x, 0, z, x, p.h, z);
  });
  const sg = new THREE.BufferGeometry();
  sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(stemPos), 3));
  stems.geometry.dispose();
  stems.geometry = sg;
  stems.computeLineDistances();
}

let viewFramed = false;
function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (!viewFramed) { viewFramed = true; setPerspectiveView(); }
}
new ResizeObserver(resize).observe(viewport);
new ResizeObserver(() => {
  fitCanvas(normalCanvas);
  fitCanvas(litCanvas);
  if (preview) positionLightMarker();
}).observe(litFrame);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

// --- Pointer interaction ---------------------------------------------------

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let drag = null;   // active vertex drag
let press = null;  // possible click on empty space / surface

function setRay(e) {
  const r = canvasEl.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
}

function onPointerDown(e) {
  if (e.button !== 0) return;
  setRay(e);
  const hit = raycaster.intersectObjects(handles.children, false)[0];
  if (hit) {
    const i = hit.object.userData.index;
    const v = state.verts[i];
    select(i);
    let mode = state.dragMode;
    if (e.shiftKey) mode = mode === 'height' ? 'move' : 'height';
    if (v.corner) mode = 'height';
    const r = canvasEl.getBoundingClientRect();
    const dist = camera.position.distanceTo(hit.object.position);
    drag = {
      i, mode,
      startY: e.clientY,
      startH: v.h,
      unitsPerPx: (2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / r.height,
    };
    controls.enabled = false;
    canvasEl.setPointerCapture(e.pointerId);
    e.preventDefault();
    e.stopImmediatePropagation(); // keep OrbitControls from starting an orbit
    return;
  }
  press = { x: e.clientX, y: e.clientY };
}

const tmpPlane = new THREE.Plane();
const tmpVec = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);

function onPointerMove(e) {
  if (!drag) return;
  const v = state.verts[drag.i];
  if (drag.mode === 'height') {
    const h = drag.startH + (drag.startY - e.clientY) * drag.unitsPerPx;
    v.h = Math.round(h * 1000) / 1000;
  } else {
    setRay(e);
    tmpPlane.set(UP, -v.h);
    if (raycaster.ray.intersectPlane(tmpPlane, tmpVec)) {
      const { u, v: vv } = toUV(tmpVec.x, tmpVec.z);
      const nu = clamp(u, 0, 1), nv = clamp(vv, 0, 1);
      if (!tooClose(nu, nv, drag.i)) { v.u = nu; v.v = nv; }
    }
  }
  scheduleGeometry();
  updateSelectionPanel();
}

function onPointerUp(e) {
  if (drag) {
    drag = null;
    controls.enabled = true;
    if (canvasEl.hasPointerCapture(e.pointerId)) canvasEl.releasePointerCapture(e.pointerId);
    return;
  }
  if (press && e.type === 'pointerup' && e.button === 0) {
    const moved = Math.hypot(e.clientX - press.x, e.clientY - press.y);
    if (moved < 5) {
      setRay(e);
      const hit = raycaster.intersectObject(surface, false)[0];
      if (hit) addVertexAt(hit.point.x, hit.point.z, hit.point.y);
      else select(-1);
    }
  }
  press = null;
}

function tooClose(u, v, ignore = -1) {
  const { w, d } = planeDims();
  return state.verts.some((p, i) => i !== ignore && Math.hypot((p.u - u) * w, (p.v - v) * d) < MIN_VERTEX_GAP * PLANE_LONG);
}

function addVertexAt(x, z, h) {
  const { u, v } = toUV(x, z);
  const nu = clamp(u, 0, 1), nv = clamp(v, 0, 1);
  if (tooClose(nu, nv)) return;
  state.verts.push({ u: nu, v: nv, h: Math.round(h * 1000) / 1000 });
  select(state.verts.length - 1);
  scheduleGeometry();
}

function deleteSelected() {
  if (selected < 0) return;
  const v = state.verts[selected];
  if (v.corner) { v.h = 0; } // corners can't be removed; reset them instead
  else { state.verts.splice(selected, 1); selected = -1; }
  updateSelectionPanel();
  scheduleGeometry();
}

window.addEventListener('keydown', (e) => {
  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); deleteSelected(); }
  if (e.key === 'Escape') select(-1);
});

// ---------------------------------------------------------------------------
// Selection panel
// ---------------------------------------------------------------------------

function select(i) {
  selected = i;
  updateSelectionPanel();
  // Recolor handles without a full rebuild.
  handles.children.forEach((m) => {
    const p = state.verts[m.userData.index];
    m.material = m.userData.index === selected ? handleMats.selected : p.corner ? handleMats.corner : handleMats.normal;
  });
}

function fmt(n, digits = 3) { return String(Math.round(n * 10 ** digits) / 10 ** digits); }

function updateSelectionPanel() {
  const v = state.verts[selected];
  $('sel-empty').hidden = !!v;
  $('sel-fields').hidden = !v;
  if (!v) return;
  const setIfIdle = (id, val) => { const el = $(id); if (document.activeElement !== el) el.value = val; };
  setIfIdle('sel-x', fmt(v.u * state.texW, 2));
  setIfIdle('sel-y', fmt(v.v * state.texH, 2));
  setIfIdle('sel-h', fmt(v.h));
  $('sel-x').disabled = $('sel-y').disabled = !!v.corner;
  $('sel-corner-note').hidden = !v.corner;
  $('btn-delete').textContent = v.corner ? 'Reset height to 0' : 'Delete vertex';
}

function bindSelField(id, apply) {
  $(id).addEventListener('input', (e) => {
    const v = state.verts[selected];
    const n = parseFloat(e.target.value);
    if (!v || !Number.isFinite(n)) return;
    apply(v, n);
    scheduleGeometry();
  });
  $(id).addEventListener('change', updateSelectionPanel);
}
bindSelField('sel-x', (v, n) => { const u = clamp(n / state.texW, 0, 1); if (!tooClose(u, v.v, selected)) v.u = u; });
bindSelField('sel-y', (v, n) => { const vv = clamp(n / state.texH, 0, 1); if (!tooClose(v.u, vv, selected)) v.v = vv; });
bindSelField('sel-h', (v, n) => { v.h = n; });
$('btn-delete').addEventListener('click', deleteSelected);

function updateStats() {
  $('stats').textContent = `${state.verts.length} vertices · ${mesh.triangles.length / 3} triangles`;
}

// ---------------------------------------------------------------------------
// Texture settings & export
// ---------------------------------------------------------------------------

function setTexSize(w, h) {
  state.texW = clamp(Math.round(w) || 1, 1, TEX_MAX);
  state.texH = clamp(Math.round(h) || 1, 1, TEX_MAX);
  $('tex-w').value = state.texW;
  $('tex-h').value = state.texH;
  // Keep the current viewing angle, but move in/out so the resized plane fits.
  const offset = camera.position.clone().sub(controls.target);
  camera.position.copy(controls.target).add(offset.setLength(fitDistance()));
  scheduleGeometry();
  updateSelectionPanel();
}
$('tex-w').addEventListener('change', (e) => setTexSize(+e.target.value, state.texH));
$('tex-h').addEventListener('change', (e) => setTexSize(state.texW, +e.target.value));
$('btn-swap').addEventListener('click', () => setTexSize(state.texH, state.texW));

$('convention').addEventListener('change', (e) => {
  state.convention = e.target.value;
  if (state.previewKind === 'normal') scheduleTexture(); else saveState();
});

function exportPng(kind) {
  const W = state.texW, H = state.texH;
  const r = rasterize(W, H, kind === 'height');
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(W, H);
  if (kind === 'height') encodeHeightMap(r, img.data); else encodeNormalMap(r, img.data);
  ctx.putImageData(img, 0, 0);
  const suffix = kind === 'height' ? 'height' : `normal_${state.convention}`;
  c.toBlob((blob) => download(blob, `${suffix}_${W}x${H}.png`), 'image/png');
}
$('btn-export-normal').addEventListener('click', () => exportPng('normal'));
$('btn-export-height').addEventListener('click', () => exportPng('height'));

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// Project save/load
$('btn-save-json').addEventListener('click', () => {
  download(new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' }), 'normal-map-project.json');
});
$('btn-load-json').addEventListener('click', () => $('file-json').click());
$('file-json').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    const s = JSON.parse(await file.text());
    if (!isValidState(s)) throw new Error('not a Normal Map Builder project');
    applyState({ ...defaultState(), ...s });
  } catch (err) {
    alert(`Could not load that file: ${err.message}`);
  }
});
$('btn-reset').addEventListener('click', () => {
  if (confirm('Discard all vertices and settings?')) applyState(defaultState());
});

// ---------------------------------------------------------------------------
// Lighting controls
// ---------------------------------------------------------------------------

function syncLightControls() {
  $('light-type').value = state.light.type;
  $('light-height').value = state.light.height;
  $('light-height-out').textContent = fmt(state.light.height, 1);
  $('ambient').value = state.light.ambient;
  $('ambient-out').textContent = `${Math.round(state.light.ambient * 100)}%`;
}
$('light-type').addEventListener('change', (e) => { state.light.type = e.target.value; scheduleLit(); saveState(); });
$('light-height').addEventListener('input', (e) => { state.light.height = +e.target.value; syncLightControls(); scheduleLit(); saveState(); });
$('ambient').addEventListener('input', (e) => { state.light.ambient = +e.target.value; syncLightControls(); scheduleLit(); saveState(); });

// ---------------------------------------------------------------------------
// Gradient editor
// ---------------------------------------------------------------------------

const gradBar = $('grad-bar');
const gradMarkers = $('grad-markers');

function gradientChanged() {
  renderGradientUI();
  scheduleLit();
  saveState();
}

function renderGradientBar() {
  const ctx = gradBar.getContext('2d');
  const lut = buildLut();
  const img = ctx.createImageData(256, 1);
  for (let i = 0; i < 256; i++) {
    img.data[4 * i] = lut[3 * i]; img.data[4 * i + 1] = lut[3 * i + 1]; img.data[4 * i + 2] = lut[3 * i + 2];
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function renderGradientUI() {
  const g = state.gradient;
  document.querySelectorAll('#grad-mode button').forEach((b) => b.classList.toggle('active', b.dataset.v === g.mode));
  $('smooth-editor').hidden = g.mode !== 'smooth';
  $('stepped-editor').hidden = g.mode !== 'stepped';
  gradBar.style.cursor = g.mode === 'smooth' ? 'copy' : 'default';
  renderGradientBar();

  // Stop markers (smooth mode only)
  gradMarkers.replaceChildren();
  if (g.mode === 'smooth') {
    selectedStop = clamp(selectedStop, 0, g.stops.length - 1);
    g.stops.forEach((s, i) => {
      const m = document.createElement('div');
      m.className = 'stop-marker' + (i === selectedStop ? ' selected' : '');
      m.style.left = `${s.p * 100}%`;
      m.style.background = s.c;
      m.addEventListener('pointerdown', (e) => startStopDrag(e, i));
      gradMarkers.appendChild(m);
    });
    const s = g.stops[selectedStop];
    if (document.activeElement !== $('stop-color')) $('stop-color').value = s.c;
    if (document.activeElement !== $('stop-pos')) $('stop-pos').value = Math.round(s.p * 100);
    $('btn-stop-delete').disabled = g.stops.length <= 2;
  } else {
    $('step-count').value = g.steps.length;
    renderStepList();
  }
}

function renderStepList() {
  const list = $('step-list');
  const steps = state.gradient.steps;
  const n = steps.length;
  // Rebuild only when the count changes so open color pickers stay open.
  if (list.children.length !== n) {
    list.replaceChildren();
    steps.forEach((c, i) => {
      const label = document.createElement('label');
      label.textContent = `${Math.round((i / n) * 100)}–${Math.round(((i + 1) / n) * 100)}%`;
      const input = document.createElement('input');
      input.type = 'color';
      input.addEventListener('input', (e) => {
        state.gradient.steps[i] = e.target.value;
        renderGradientBar();
        scheduleLit();
        saveState();
      });
      label.appendChild(input);
      list.appendChild(label);
    });
  }
  [...list.querySelectorAll('input')].forEach((inp, i) => { if (inp.value !== steps[i]) inp.value = steps[i]; });
}

function barPos(e) {
  const r = gradBar.getBoundingClientRect();
  return clamp((e.clientX - r.left) / r.width, 0, 1);
}

function startStopDrag(e, i) {
  e.preventDefault();
  selectedStop = i;
  const el = e.currentTarget;
  el.setPointerCapture(e.pointerId);
  renderGradientUI();
  const marker = gradMarkers.children[i];
  const move = (ev) => {
    state.gradient.stops[i].p = Math.round(barPos(ev) * 1000) / 1000;
    marker.style.left = `${state.gradient.stops[i].p * 100}%`;
    $('stop-pos').value = Math.round(state.gradient.stops[i].p * 100);
    renderGradientBar();
    scheduleLit();
  };
  const up = () => {
    marker.removeEventListener('pointermove', move);
    marker.removeEventListener('pointerup', up);
    marker.removeEventListener('pointercancel', up);
    gradientChanged();
  };
  marker.addEventListener('pointermove', move);
  marker.addEventListener('pointerup', up);
  marker.addEventListener('pointercancel', up);
}

gradBar.addEventListener('click', (e) => {
  if (state.gradient.mode !== 'smooth') return;
  const p = Math.round(barPos(e) * 1000) / 1000;
  state.gradient.stops.push({ p, c: rgbToHex(sampleGradient(p)) });
  selectedStop = state.gradient.stops.length - 1;
  gradientChanged();
});

$('stop-color').addEventListener('input', (e) => {
  state.gradient.stops[selectedStop].c = e.target.value;
  gradientChanged();
});
$('stop-pos').addEventListener('input', (e) => {
  const n = parseFloat(e.target.value);
  if (!Number.isFinite(n)) return;
  state.gradient.stops[selectedStop].p = clamp(n / 100, 0, 1);
  gradientChanged();
});
$('btn-stop-delete').addEventListener('click', () => {
  if (state.gradient.stops.length <= 2) return;
  state.gradient.stops.splice(selectedStop, 1);
  selectedStop = Math.max(0, selectedStop - 1);
  gradientChanged();
});

function resampleSteps(n) {
  state.gradient.steps = Array.from({ length: n }, (_, i) => rgbToHex(sampleGradient(n === 1 ? 0 : i / (n - 1))));
}

document.querySelectorAll('#grad-mode button').forEach((b) => b.addEventListener('click', () => {
  state.gradient.mode = b.dataset.v;
  gradientChanged();
}));
$('step-count').addEventListener('change', (e) => {
  const n = clamp(Math.round(+e.target.value) || 2, 2, 16);
  if (n !== state.gradient.steps.length) resampleSteps(n);
  gradientChanged();
});
$('btn-resample').addEventListener('click', () => { resampleSteps(state.gradient.steps.length); gradientChanged(); });

// ---------------------------------------------------------------------------
// Misc UI wiring
// ---------------------------------------------------------------------------

function syncSegs() {
  document.querySelectorAll('#drag-mode button').forEach((b) => b.classList.toggle('active', b.dataset.v === state.dragMode));
  document.querySelectorAll('#preview-kind button').forEach((b) => b.classList.toggle('active', b.dataset.v === state.previewKind));
}
document.querySelectorAll('#drag-mode button').forEach((b) => b.addEventListener('click', () => {
  state.dragMode = b.dataset.v; syncSegs(); saveState();
}));
document.querySelectorAll('#preview-kind button').forEach((b) => b.addEventListener('click', () => {
  state.previewKind = b.dataset.v; syncSegs(); scheduleTexture();
}));
$('btn-top').addEventListener('click', setTopView);
$('btn-persp').addEventListener('click', setPerspectiveView);

function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }

function applyState(s) {
  state = s;
  selected = -1;
  selectedStop = 0;
  $('tex-w').value = state.texW;
  $('tex-h').value = state.texH;
  $('convention').value = state.convention;
  syncSegs();
  syncLightControls();
  renderGradientUI();
  updateSelectionPanel();
  scheduleGeometry();
}

applyState(state);
