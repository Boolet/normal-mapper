import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Delaunator from 'delaunator';
import { createGizmo, RING_RADIUS } from './gizmo.js';
import { buildConeLut, sampleCone, bendDiffuse, lutImagePixels, LUT_W, LUT_H } from './shading.js';

// ---------------------------------------------------------------------------
// Constants & state
// ---------------------------------------------------------------------------

const PLANE_LONG = 10;        // world length of the plane's longer side
const PREVIEW_MAX = 512;      // longest side of the on-screen previews, in pixels
const TEX_MAX = 4096;         // largest exportable texture side
const PIXEL_MAX = 256;        // largest texture side in pixel-perfect mode
const H_MIN = -1, H_MAX = 1;  // vertex height limits (before the height scale)
const MIN_VERTEX_GAP = 1e-3;  // fraction of the plane; closer freeform vertices are rejected
const PICK_RADIUS_PX = 9;     // screen distance for picking a grid vertex
const UNDO_LIMIT = 60;
const MAX_TILT = THREE.MathUtils.degToRad(85);
const STORAGE_KEY = 'normal-map-builder/v1';
const CURV_KEY = 'normal-map-builder/v1/curvature';

const $ = (id) => document.getElementById(id);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

function defaultState() {
  return {
    texW: 512,
    texH: 512,
    convention: 'opengl',
    previewKind: 'normal',
    mode: 'free',            // 'free' | 'pixel'
    dragMode: 'height',
    heightScale: 2,
    snapPivot: true,
    tool2d: 'select',        // 'select' | 'paint'
    brush: { size: 1, value: 255 },
    verts: [
      { u: 0, v: 0, h: 0, corner: true },
      { u: 1, v: 0, h: 0, corner: true },
      { u: 1, v: 1, h: 0, corner: true },
      { u: 0, v: 1, h: 0, corner: true },
      { u: 0.5, v: 0.5, h: 1 },
    ],
    grid: null,              // serialized pixel grid; the live copy is `grid` below
    light: { u: 0.2, v: 0.2, height: 6, type: 'point', ambient: 0.05, curveModel: 'bend' },
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

/** Fill in missing fields and migrate projects saved before height limits existed. */
function normalizeState(s) {
  const d = defaultState();
  const out = { ...d, ...s, light: { ...d.light, ...s.light }, brush: { ...d.brush, ...(s.brush || {}) } };
  if (!Number.isFinite(s.heightScale)) {
    // Old projects stored raw heights: fold their range into the scale.
    const maxAbs = Math.max(0, ...out.verts.map((v) => Math.abs(v.h)));
    out.heightScale = maxAbs > 1 ? maxAbs : 1;
    if (maxAbs > 1) out.verts = out.verts.map((v) => ({ ...v, h: v.h / maxAbs }));
  }
  out.heightScale = clamp(out.heightScale, 0.01, 100);
  out.verts = out.verts.map((v) => ({ ...v, h: clamp(v.h, H_MIN, H_MAX) }));
  out.texW = clamp(Math.round(out.texW), 1, TEX_MAX);
  out.texH = clamp(Math.round(out.texH), 1, TEX_MAX);
  if (out.mode !== 'pixel' || Math.max(out.texW, out.texH) > PIXEL_MAX) out.mode = 'free';
  return out;
}

function gridFromSerialized(g) {
  if (!g || !(g.W >= 1) || !(g.H >= 1) || !Array.isArray(g.h) || g.h.length !== (g.W + 1) * (g.H + 1)) return null;
  return { W: g.W, H: g.H, h: Float32Array.from(g.h, (x) => clamp(+x || 0, H_MIN, H_MAX)) };
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    return isValidState(s) ? s : null;
  } catch { return null; }
}

function serialize() {
  return {
    ...state,
    grid: grid ? { W: grid.W, H: grid.H, h: Array.from(grid.h, (x) => Math.round(x * 1e4) / 1e4) } : null,
  };
}

let saveTimer = 0;
function saveState() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(serialize())); } catch { /* storage unavailable */ }
  }, 300);
}

let state = defaultState();
let grid = null;            // { W, H, h: Float32Array((W+1)*(H+1)) } — pixel-corner heights
let curv = { W: 1, H: 1, data: new Uint8Array(1) }; // curvature map at texture resolution
let selected = -1;          // freeform selection
let limitVertex = -1;       // freeform vertex that hit a height limit
let selectedStop = 0;

// Pixel-mode selection
let sel = new Uint8Array(0);
let selCount = 0;
let limitIdx = -1;          // grid vertex that stopped the last group transform
const pivot = { x: 0, z: 0 };
let fitCache = null;

// ---------------------------------------------------------------------------
// Plane geometry helpers
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

// ---------------------------------------------------------------------------
// Freeform surface: triangulation & rasterization
// ---------------------------------------------------------------------------

let mesh = { triangles: new Uint32Array(0), normals: new Float32Array(0) };

function triangulate() {
  const { w, d } = planeDims();
  const s = state.heightScale;
  const V = state.verts;
  const coords = new Float64Array(V.length * 2);
  V.forEach((p, i) => { coords[2 * i] = p.u * w; coords[2 * i + 1] = p.v * d; });
  const triangles = new Delaunator(coords).triangles;
  const n = triangles.length / 3;
  const normals = new Float32Array(n * 3);
  for (let t = 0; t < n; t++) {
    const a = V[triangles[3 * t]], b = V[triangles[3 * t + 1]], c = V[triangles[3 * t + 2]];
    const ax = a.u * w, az = a.v * d;
    const e1x = b.u * w - ax, e1y = (b.h - a.h) * s, e1z = b.v * d - az;
    const e2x = c.u * w - ax, e2y = (c.h - a.h) * s, e2z = c.v * d - az;
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
 * Rasterize the freeform surface into a W×H grid sampled at pixel centers.
 * Returns the covering triangle per pixel and optionally the interpolated
 * (unscaled) height.
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

/** Unscaled freeform height at (u, v), used to seed the pixel grid. */
function sampleFreeform(u, v) {
  const V = state.verts;
  const T = mesh.triangles;
  let best = 0, bestErr = Infinity;
  for (let t = 0; t < T.length / 3; t++) {
    const A = V[T[3 * t]], B = V[T[3 * t + 1]], C = V[T[3 * t + 2]];
    const area = (B.u - A.u) * (C.v - A.v) - (B.v - A.v) * (C.u - A.u);
    if (Math.abs(area) < 1e-14) continue;
    const w0 = ((B.u - u) * (C.v - v) - (B.v - v) * (C.u - u)) / area;
    const w1 = ((C.u - u) * (A.v - v) - (C.v - v) * (A.u - u)) / area;
    const w2 = 1 - w0 - w1;
    const err = Math.max(0, -w0, -w1, -w2);
    if (err < bestErr) { bestErr = err; best = w0 * A.h + w1 * B.h + w2 * C.h; }
    if (err === 0) break;
  }
  return clamp(best, H_MIN, H_MAX);
}

// ---------------------------------------------------------------------------
// Pixel-perfect surface
// ---------------------------------------------------------------------------

function gridFromFreeform(W, H) {
  triangulate();
  const h = new Float32Array((W + 1) * (H + 1));
  for (let j = 0; j <= H; j++) for (let i = 0; i <= W; i++) h[j * (W + 1) + i] = sampleFreeform(i / W, j / H);
  return { W, H, h };
}

/** Bilinearly resample a corner grid to new pixel dimensions. */
function resampleGrid(g, W, H) {
  const h = new Float32Array((W + 1) * (H + 1));
  for (let j = 0; j <= H; j++) {
    const fy = (j / H) * g.H, y0 = Math.min(g.H - 1, Math.floor(fy)), ty = fy - y0;
    for (let i = 0; i <= W; i++) {
      const fx = (i / W) * g.W, x0 = Math.min(g.W - 1, Math.floor(fx)), tx = fx - x0;
      const at = (x, y) => g.h[y * (g.W + 1) + x];
      const top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * tx;
      const bot = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * tx;
      h[j * (W + 1) + i] = top + (bot - top) * ty;
    }
  }
  return { W, H, h };
}

function gridWorld(idx) {
  const W1 = grid.W + 1;
  const i = idx % W1, j = (idx - i) / W1;
  const { w, d } = planeDims();
  return { x: (i / grid.W - 0.5) * w, z: (j / grid.H - 0.5) * d };
}

let pixelSurf = null; // { n: Float32Array(3WH), hc: Float32Array(WH) (world height at pixel centers) }

/** One normal per pixel: the average plane of its four corners. */
function computePixelSurface() {
  const { W, H, h } = grid;
  const { w, d } = planeDims();
  const s = state.heightScale;
  const pw = w / W, ph = d / H, W1 = W + 1;
  const n = new Float32Array(3 * W * H);
  const hc = new Float32Array(W * H);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const a = h[j * W1 + i], b = h[j * W1 + i + 1], dd = h[(j + 1) * W1 + i], c = h[(j + 1) * W1 + i + 1];
      const gx = ((b + c) - (a + dd)) * s / (2 * pw);
      const gz = ((dd + c) - (a + b)) * s / (2 * ph);
      const len = Math.hypot(gx, 1, gz);
      const k = j * W + i;
      n[3 * k] = -gx / len;
      n[3 * k + 1] = 1 / len;
      n[3 * k + 2] = -gz / len;
      hc[k] = ((a + b + c + dd) / 4) * s;
    }
  }
  pixelSurf = { n, hc };
}

// ---------------------------------------------------------------------------
// Curvature map
// ---------------------------------------------------------------------------

function resampleCurvature(W, H) {
  if (curv.W === W && curv.H === H) return;
  const data = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const sy = Math.min(curv.H - 1, Math.floor(((y + 0.5) / H) * curv.H));
    for (let x = 0; x < W; x++) {
      const sx = Math.min(curv.W - 1, Math.floor(((x + 0.5) / W) * curv.W));
      data[y * W + x] = curv.data[sy * curv.W + sx];
    }
  }
  curv = { W, H, data };
}

function curvatureCanvas() {
  const c = document.createElement('canvas');
  c.width = curv.W; c.height = curv.H;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(curv.W, curv.H);
  for (let k = 0; k < curv.data.length; k++) {
    img.data[4 * k] = img.data[4 * k + 1] = img.data[4 * k + 2] = curv.data[k];
    img.data[4 * k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

const curvIsEmpty = () => !curv.data.some((x) => x !== 0);
const curvDataUrl = () => (curvIsEmpty() ? null : curvatureCanvas().toDataURL('image/png'));

/** Load an image (URL or data URL) into the curvature map, scaled to the texture size. */
function loadCurvatureImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const W = state.texW, H = state.texH;
      const c = document.createElement('canvas');
      c.width = W; c.height = H;
      const ctx = c.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, W, H);
      const px = ctx.getImageData(0, 0, W, H).data;
      const data = new Uint8Array(W * H);
      for (let k = 0; k < W * H; k++) {
        data[k] = Math.round(0.2126 * px[4 * k] + 0.7152 * px[4 * k + 1] + 0.0722 * px[4 * k + 2]);
      }
      curv = { W, H, data };
      resolve();
    };
    img.onerror = () => reject(new Error('could not decode the image'));
    img.src = src;
  });
}

let curvSaveTimer = 0;
function saveCurvature() {
  clearTimeout(curvSaveTimer);
  curvSaveTimer = setTimeout(() => {
    try {
      const url = curvDataUrl();
      if (url) localStorage.setItem(CURV_KEY, url); else localStorage.removeItem(CURV_KEY);
    } catch { /* storage unavailable or full */ }
  }, 500);
}

// ---------------------------------------------------------------------------
// Undo / redo (surface, texture size, height scale and curvature)
// ---------------------------------------------------------------------------

const undoStack = [];
const redoStack = [];

function snapshot() {
  return {
    verts: structuredClone(state.verts),
    grid: grid ? { W: grid.W, H: grid.H, h: grid.h.slice() } : null,
    heightScale: state.heightScale,
    texW: state.texW,
    texH: state.texH,
    curv: { W: curv.W, H: curv.H, data: curv.data.slice() },
  };
}

function pushUndo() {
  undoStack.push(snapshot());
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  redoStack.length = 0;
  updateUndoButtons();
}

function restoreSnapshot(s) {
  state.verts = s.verts;
  state.heightScale = s.heightScale;
  state.texW = s.texW;
  state.texH = s.texH;
  grid = s.grid;
  curv = s.curv;
  if (state.mode === 'pixel' && (!grid || Math.max(state.texW, state.texH) > PIXEL_MAX)) state.mode = 'free';
  selected = -1;
  limitVertex = -1;
  resetSelection();
  syncAllUI();
  scheduleGeometry();
  scheduleCurvature();
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(snapshot());
  restoreSnapshot(undoStack.pop());
  updateUndoButtons();
}
function redo() {
  if (!redoStack.length) return;
  undoStack.push(snapshot());
  restoreSnapshot(redoStack.pop());
  updateUndoButtons();
}
function updateUndoButtons() {
  $('btn-undo').disabled = !undoStack.length;
  $('btn-redo').disabled = !redoStack.length;
}
$('btn-undo').addEventListener('click', undo);
$('btn-redo').addEventListener('click', redo);

/** For number inputs: snapshot once per editing session, not on every keystroke. */
function undoOnFirstInput(el) {
  el.addEventListener('input', () => {
    if (!el.dataset.editing) { pushUndo(); el.dataset.editing = '1'; }
  });
  const end = () => { delete el.dataset.editing; };
  el.addEventListener('change', end);
  el.addEventListener('blur', end);
}

// ---------------------------------------------------------------------------
// Texture encoding
// ---------------------------------------------------------------------------

function writeNormal(data, k, nx, ny, nz, flipG) {
  // Tangent space: R = +X (right), G = up in the image (−Z for OpenGL), B = surface normal (+Y).
  const g = flipG ? nz : -nz;
  data[4 * k] = Math.round((nx * 0.5 + 0.5) * 255);
  data[4 * k + 1] = Math.round((g * 0.5 + 0.5) * 255);
  data[4 * k + 2] = Math.round((ny * 0.5 + 0.5) * 255);
  data[4 * k + 3] = 255;
}

function encodeNormalArray(n, count, data) {
  const flipG = state.convention === 'directx';
  for (let k = 0; k < count; k++) writeNormal(data, k, n[3 * k], n[3 * k + 1], n[3 * k + 2], flipG);
}

function encodeNormalTri(r, data) {
  const N = mesh.normals;
  const flipG = state.convention === 'directx';
  for (let k = 0, n = r.W * r.H; k < n; k++) {
    const t = r.tri[k];
    if (t >= 0) writeNormal(data, k, N[3 * t], N[3 * t + 1], N[3 * t + 2], flipG);
    else writeNormal(data, k, 0, 1, 0, flipG);
  }
}

/** World heights → gray, mapping the −1…1 limits (times the scale) to 0…255. */
function encodeHeight(hw, count, data) {
  const s = state.heightScale;
  for (let k = 0; k < count; k++) {
    const g = Math.round(clamp((hw[k] / s + 1) / 2, 0, 1) * 255);
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
const overlay = $('tex-overlay');
const texFrame = $('tex-frame');
const litCanvas = $('lit-canvas');
const litFrame = $('lit-frame');
const lightMarker = $('light-marker');
const coneLut = buildConeLut();
let preview = null; // { W, H, n, hw } at preview resolution

function previewSize() {
  const s = Math.min(1, PREVIEW_MAX / Math.max(state.texW, state.texH));
  return { W: Math.max(1, Math.round(state.texW * s)), H: Math.max(1, Math.round(state.texH * s)) };
}

const PREVIEW_MAX_CSS_HEIGHT = 340;
function sizeCanvas(c, W, H) {
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  c.classList.toggle('pixelated', Math.max(W, H) < 256);
  fitCanvas(c);
}
// Scale to the frame's width without exceeding the max height, keeping the aspect ratio.
function fitCanvas(c) {
  const frameW = c.parentElement.clientWidth - 2;
  if (frameW <= 0) return;
  const s = Math.min(frameW / c.width, PREVIEW_MAX_CSS_HEIGHT / c.height);
  c.style.width = `${c.width * s}px`;
  c.style.height = `${c.height * s}px`;
}

/** Per-pixel normals and world heights for the active surface. */
function surfaceBuffers() {
  const { W, H } = previewSize();
  if (state.mode === 'pixel') return { W, H, n: pixelSurf.n, hw: pixelSurf.hc };
  const r = rasterize(W, H, true);
  const N = mesh.normals;
  const s = state.heightScale;
  const n = new Float32Array(3 * W * H);
  const hw = new Float32Array(W * H);
  for (let k = 0; k < W * H; k++) {
    const t = r.tri[k];
    if (t >= 0) { n[3 * k] = N[3 * t]; n[3 * k + 1] = N[3 * t + 1]; n[3 * k + 2] = N[3 * t + 2]; } else n[3 * k + 1] = 1;
    hw[k] = r.height[k] * s;
  }
  return { W, H, n, hw };
}

function updatePreviews() {
  preview = surfaceBuffers();
  sizeCanvas(normalCanvas, preview.W, preview.H);
  sizeCanvas(litCanvas, preview.W, preview.H);
  drawTexturePreview();
  drawLit();
}

function drawTexturePreview() {
  const { W, H } = preview;
  const ctx = normalCanvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  if (state.previewKind === 'height') encodeHeight(preview.hw, W * H, img.data);
  else if (state.previewKind === 'curvature') {
    for (let y = 0; y < H; y++) {
      const cy = Math.min(curv.H - 1, Math.floor(((y + 0.5) / H) * curv.H));
      for (let x = 0; x < W; x++) {
        const cx = Math.min(curv.W - 1, Math.floor(((x + 0.5) / W) * curv.W));
        const k = y * W + x, g = curv.data[cy * curv.W + cx];
        img.data[4 * k] = img.data[4 * k + 1] = img.data[4 * k + 2] = g;
        img.data[4 * k + 3] = 255;
      }
    }
  } else encodeNormalArray(preview.n, W * H, img.data);
  ctx.putImageData(img, 0, 0);
  drawOverlay();
}

function drawLit() {
  const { W, H, n: N, hw } = preview;
  const { w, d } = planeDims();
  const L = state.light;
  const lx = (L.u - 0.5) * w, lz = (L.v - 0.5) * d, ly = L.height;
  const amb = L.ambient;
  const point = L.type === 'point';
  const cone = L.curveModel === 'cone';
  // A directional light shines from the plane's center toward the marker.
  let dx = lx, dy = ly, dz = lz;
  if (!point) { const len = Math.hypot(dx, dy, dz) || 1; dx /= len; dy /= len; dz /= len; }
  const lut = buildLut();
  const cW = curv.W, cH = curv.H, cData = curv.data;
  const ctx = litCanvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  const data = img.data;
  for (let py = 0; py < H; py++) {
    const z = ((py + 0.5) / H - 0.5) * d;
    const crow = Math.min(cH - 1, Math.floor(((py + 0.5) / H) * cH)) * cW;
    for (let px = 0; px < W; px++) {
      const k = py * W + px;
      const nx = N[3 * k], ny = N[3 * k + 1], nz = N[3 * k + 2];
      let ndl;
      if (point) {
        const x = ((px + 0.5) / W - 0.5) * w;
        const vx = lx - x, vy = ly - hw[k], vz = lz - z;
        ndl = (nx * vx + ny * vy + nz * vz) / (Math.hypot(vx, vy, vz) || 1);
      } else {
        ndl = nx * dx + ny * dy + nz * dz;
      }
      const c = cData[crow + Math.min(cW - 1, Math.floor(((px + 0.5) / W) * cW))] / 255;
      const diff = c > 0 ? (cone ? sampleCone(coneLut, ndl, c) : bendDiffuse(ndl, c)) : (ndl > 0 ? ndl : 0);
      const I = amb + (1 - amb) * diff;
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

// Selection dots and the drag rectangle, drawn over the texture preview.
let texDragRect = null; // { x0, y0, x1, y1, sub } in texture pixels
function drawOverlay() {
  const cssW = normalCanvas.clientWidth, cssH = normalCanvas.clientHeight;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  overlay.style.left = `${normalCanvas.offsetLeft}px`;
  overlay.style.top = `${normalCanvas.offsetTop}px`;
  overlay.style.width = `${cssW}px`;
  overlay.style.height = `${cssH}px`;
  const bw = Math.max(1, Math.round(cssW * dpr)), bh = Math.max(1, Math.round(cssH * dpr));
  if (overlay.width !== bw || overlay.height !== bh) { overlay.width = bw; overlay.height = bh; }
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, bw, bh);
  if (state.mode !== 'pixel' || !grid) return;
  const sx = bw / grid.W, sy = bh / grid.H;
  if (selCount) {
    const r = clamp(Math.min(sx, sy) * 0.18, 1.5 * dpr, 3.5 * dpr);
    ctx.fillStyle = '#4dd8ff';
    ctx.strokeStyle = 'rgba(0,0,0,.7)';
    ctx.lineWidth = dpr;
    const W1 = grid.W + 1;
    ctx.beginPath();
    for (let idx = 0; idx < sel.length; idx++) {
      if (!sel[idx]) continue;
      const i = idx % W1, j = (idx - i) / W1;
      ctx.moveTo(i * sx + r, j * sy);
      ctx.arc(i * sx, j * sy, r, 0, Math.PI * 2);
    }
    ctx.fill();
    ctx.stroke();
  }
  if (texDragRect) {
    const { x0, y0, x1, y1, sub } = texDragRect;
    ctx.strokeStyle = sub ? '#ff6b7d' : '#5aa9ff';
    ctx.lineWidth = 1.5 * dpr;
    ctx.setLineDash([4 * dpr, 3 * dpr]);
    ctx.strokeRect(Math.min(x0, x1) * sx, Math.min(y0, y1) * sy, (Math.abs(x1 - x0) + 1) * sx, (Math.abs(y1 - y0) + 1) * sy);
    ctx.setLineDash([]);
  }
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

let pending = { geometry: false, lit: false, texture: false, curvature: false, selection: false };
let frameQueued = false;
function queueFrame() {
  if (frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => {
    frameQueued = false;
    const p = pending;
    pending = { geometry: false, lit: false, texture: false, curvature: false, selection: false };
    if (p.geometry) {
      if (state.mode === 'pixel') {
        computePixelSurface();
        updatePixelScene();
        fitCache = null;
      } else {
        triangulate();
        rebuildFreeScene();
      }
      updatePreviews();
      updateStats();
    } else {
      if (p.texture || (p.curvature && state.previewKind === 'curvature')) drawTexturePreview();
      if (p.lit || p.curvature) drawLit();
    }
    if (p.selection) {
      updatePointColors();
      updatePixelPanel();
      drawOverlay();
    }
  });
}
function scheduleGeometry() { pending.geometry = true; queueFrame(); saveState(); }
function scheduleLit() { pending.lit = true; queueFrame(); }
function scheduleTexture() { pending.texture = true; queueFrame(); saveState(); }
function scheduleCurvature() { pending.curvature = true; queueFrame(); saveCurvature(); }
function scheduleSelection() { pending.selection = true; queueFrame(); }

// ---------------------------------------------------------------------------
// 3D editor: scene
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

// Our pointer handlers are registered before OrbitControls so they can claim
// drags that start on a handle, the gizmo, or a box selection.
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
  const r = (Math.hypot(w, d) / 2) * 1.08;
  return Math.max(r / Math.sin(halfV), r / Math.sin(halfH));
}
function setPerspectiveView() {
  const dist = fitDistance();
  // Slightly off-axis so neither transform ring is seen edge-on.
  const elev = 0.85, azim = 0.5;
  camera.position.set(
    dist * Math.cos(elev) * Math.sin(azim),
    dist * Math.sin(elev),
    dist * Math.cos(elev) * Math.cos(azim),
  );
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
const wireMat = new THREE.LineBasicMaterial({ color: 0x0c0f13, transparent: true, opacity: 0.55 });

const outline = new THREE.LineLoop(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x5aa9ff, transparent: true, opacity: 0.5 }),
);
scene.add(outline);

function updateOutline() {
  const { w, d } = planeDims();
  const og = new THREE.BufferGeometry();
  og.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
    -w / 2, 0, -d / 2, w / 2, 0, -d / 2, w / 2, 0, d / 2, -w / 2, 0, d / 2,
  ]), 3));
  outline.geometry.dispose();
  outline.geometry = og;
}

// --- Freeform objects ---
const freeGroup = new THREE.Group();
scene.add(freeGroup);
const surface = new THREE.Mesh(new THREE.BufferGeometry(), surfaceMat);
const wire = new THREE.LineSegments(new THREE.BufferGeometry(), wireMat);
const stems = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineDashedMaterial({ color: 0xffc857, dashSize: 0.12, gapSize: 0.1, transparent: true, opacity: 0.6 }),
);
const handles = new THREE.Group();
freeGroup.add(surface, wire, stems, handles);
const handleGeo = new THREE.SphereGeometry(0.14, 16, 12);
const handleMats = {
  normal: new THREE.MeshBasicMaterial({ color: 0xffc857, depthTest: false }),
  corner: new THREE.MeshBasicMaterial({ color: 0x7dd3fc, depthTest: false }),
  selected: new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }),
  limit: new THREE.MeshBasicMaterial({ color: 0xff3b55, depthTest: false }),
};
function handleMat(i) {
  if (i === limitVertex) return handleMats.limit;
  if (i === selected) return handleMats.selected;
  return state.verts[i].corner ? handleMats.corner : handleMats.normal;
}

function rebuildFreeScene() {
  const V = state.verts;
  const T = mesh.triangles;
  const s = state.heightScale;
  const pos = new Float32Array(T.length * 3);
  const edges = new Float32Array(T.length * 6);
  for (let i = 0; i < T.length; i++) {
    const p = V[T[i]];
    const { x, z } = toWorld(p.u, p.v);
    pos.set([x, p.h * s, z], 3 * i);
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

  updateOutline();

  const stemPos = [];
  handles.clear();
  V.forEach((p, i) => {
    const { x, z } = toWorld(p.u, p.v);
    const m = new THREE.Mesh(handleGeo, handleMat(i));
    m.position.set(x, p.h * s, z);
    m.renderOrder = 10;
    m.userData.index = i;
    handles.add(m);
    if (Math.abs(p.h) > 1e-4) stemPos.push(x, 0, z, x, p.h * s, z);
  });
  const sg = new THREE.BufferGeometry();
  sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(stemPos), 3));
  stems.geometry.dispose();
  stems.geometry = sg;
  stems.computeLineDistances();
}

// --- Pixel-perfect objects ---
const pixelGroup = new THREE.Group();
scene.add(pixelGroup);
const pSurface = new THREE.Mesh(new THREE.BufferGeometry(), surfaceMat);
const pWire = new THREE.LineSegments(new THREE.BufferGeometry(), wireMat);
const dotTexture = (() => {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d');
  ctx.beginPath();
  ctx.arc(32, 32, 26, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();
  ctx.lineWidth = 8;
  ctx.strokeStyle = '#000';
  ctx.stroke();
  return new THREE.CanvasTexture(c);
})();
const pointsMat = new THREE.PointsMaterial({
  size: 9, sizeAttenuation: false, vertexColors: true, map: dotTexture, alphaTest: 0.5, depthTest: false,
});
const pPoints = new THREE.Points(new THREE.BufferGeometry(), pointsMat);
pPoints.renderOrder = 10;
// Larger marker drawn over everything (including the gizmo) at a vertex that hit a height limit.
const limitMarker = new THREE.Points(
  new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3)),
  new THREE.PointsMaterial({ size: 18, sizeAttenuation: false, color: 0xff3b55, map: dotTexture, alphaTest: 0.5, depthTest: false }),
);
limitMarker.renderOrder = 30;
limitMarker.visible = false;
pixelGroup.add(pSurface, pWire, pPoints, limitMarker);
let pixelGeoDims = null;

const COLOR_POINT = new THREE.Color(0xffc857);
const COLOR_SELECTED = new THREE.Color(0x4dd8ff);
const COLOR_LIMIT = new THREE.Color(0xff3b55);

/** Write all pixel-mode geometry from the grid (allocating only when the size changes). */
function updatePixelScene() {
  const { W, H, h } = grid;
  const W1 = W + 1;
  const { w, d } = planeDims();
  const s = state.heightScale;
  const X = (i) => (i / W - 0.5) * w;
  const Z = (j) => (j / H - 0.5) * d;
  const key = `${W}x${H}:${w}x${d}`;
  if (pixelGeoDims !== key) {
    pixelGeoDims = key;
    for (const o of [pSurface, pWire, pPoints]) { o.geometry.dispose(); o.geometry = new THREE.BufferGeometry(); }
    pSurface.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(W * H * 12 * 3), 3));
    pWire.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(((H + 1) * W + (W + 1) * H) * 6), 3));
    pPoints.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(W1 * (H + 1) * 3), 3));
    pPoints.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(W1 * (H + 1) * 3), 3));
    pointsMat.size = Math.max(W, H) > 96 ? 5 : Math.max(W, H) > 48 ? 7 : 9;
    updateOutline();
  }
  const Y = (i, j) => h[j * W1 + i] * s;

  // Four triangles per pixel, meeting at the pixel center.
  const sp = pSurface.geometry.attributes.position.array;
  let o = 0;
  const put = (x, y, z) => { sp[o++] = x; sp[o++] = y; sp[o++] = z; };
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const ax = X(i), bx = X(i + 1), az = Z(j), dz = Z(j + 1);
      const ya = Y(i, j), yb = Y(i + 1, j), yc = Y(i + 1, j + 1), yd = Y(i, j + 1);
      const mx = (ax + bx) / 2, mz = (az + dz) / 2, my = (ya + yb + yc + yd) / 4;
      put(ax, ya, az); put(mx, my, mz); put(bx, yb, az);
      put(bx, yb, az); put(mx, my, mz); put(bx, yc, dz);
      put(bx, yc, dz); put(mx, my, mz); put(ax, yd, dz);
      put(ax, yd, dz); put(mx, my, mz); put(ax, ya, az);
    }
  }
  pSurface.geometry.attributes.position.needsUpdate = true;
  pSurface.geometry.computeVertexNormals();
  pSurface.geometry.computeBoundingSphere();
  pSurface.geometry.computeBoundingBox();

  // Pixel edges only (not the diagonals), so the grid reads as pixels.
  const wp = pWire.geometry.attributes.position.array;
  o = 0;
  const seg = (i0, j0, i1, j1) => {
    wp[o++] = X(i0); wp[o++] = Y(i0, j0); wp[o++] = Z(j0);
    wp[o++] = X(i1); wp[o++] = Y(i1, j1); wp[o++] = Z(j1);
  };
  for (let j = 0; j <= H; j++) for (let i = 0; i < W; i++) seg(i, j, i + 1, j);
  for (let i = 0; i <= W; i++) for (let j = 0; j < H; j++) seg(i, j, i, j + 1);
  pWire.geometry.attributes.position.needsUpdate = true;
  pWire.geometry.computeBoundingSphere();

  const pp = pPoints.geometry.attributes.position.array;
  for (let j = 0; j <= H; j++) {
    for (let i = 0; i <= W; i++) {
      const k = 3 * (j * W1 + i);
      pp[k] = X(i); pp[k + 1] = Y(i, j); pp[k + 2] = Z(j);
    }
  }
  pPoints.geometry.attributes.position.needsUpdate = true;
  pPoints.geometry.computeBoundingSphere();
  updatePointColors();
}

function updatePointColors() {
  if (!grid || !pPoints.geometry.attributes.color) return;
  const col = pPoints.geometry.attributes.color;
  if (col.count !== sel.length) return;
  for (let i = 0; i < sel.length; i++) {
    const c = i === limitIdx ? COLOR_LIMIT : sel[i] ? COLOR_SELECTED : COLOR_POINT;
    col.setXYZ(i, c.r, c.g, c.b);
  }
  col.needsUpdate = true;
  limitMarker.visible = limitIdx >= 0;
  if (limitIdx >= 0) {
    const p = gridWorld(limitIdx);
    limitMarker.geometry.attributes.position.setXYZ(0, p.x, grid.h[limitIdx] * state.heightScale, p.z);
    limitMarker.geometry.attributes.position.needsUpdate = true;
  }
}

// --- Transform gizmo ---
const gizmo = createGizmo();
gizmo.group.visible = false;
scene.add(gizmo.group);

function updateGizmoTransform() {
  const show = state.mode === 'pixel' && selCount > 0;
  gizmo.group.visible = show;
  if (!show) return;
  const y = pivotHeight();
  gizmo.group.position.set(pivot.x, y, pivot.z);
  gizmo.group.scale.setScalar(camera.position.distanceTo(gizmo.group.position) * 0.085);
}

function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (!viewFramed) { viewFramed = true; setPerspectiveView(); }
}
let viewFramed = false;
new ResizeObserver(resize).observe(viewport);
new ResizeObserver(() => {
  fitCanvas(normalCanvas);
  fitCanvas(litCanvas);
  if (preview) { positionLightMarker(); drawOverlay(); }
}).observe(litFrame);

renderer.setAnimationLoop(() => {
  controls.update();
  updateGizmoTransform();
  renderer.render(scene, camera);
});

// ---------------------------------------------------------------------------
// Pixel-mode selection & fitting
// ---------------------------------------------------------------------------

function resetSelection() {
  const n = grid ? (grid.W + 1) * (grid.H + 1) : 0;
  sel = new Uint8Array(n);
  selCount = 0;
  limitIdx = -1;
  fitCache = null;
  scheduleSelection();
}

function setSelected(idx, on) {
  if (!!sel[idx] === on) return;
  sel[idx] = on ? 1 : 0;
  selCount += on ? 1 : -1;
}

function selectionChanged() {
  fitCache = null;
  limitIdx = -1;
  resetPivot();
  scheduleSelection();
}

function selectedList() {
  const out = [];
  for (let i = 0; i < sel.length; i++) if (sel[i]) out.push(i);
  return out;
}

function snapPivotXZ(x, z) {
  const { w, d } = planeDims();
  const sx = w / grid.W / 2, sz = d / grid.H / 2;
  return {
    x: clamp(-w / 2 + Math.round((x + w / 2) / sx) * sx, -w / 2, w / 2),
    z: clamp(-d / 2 + Math.round((z + d / 2) / sz) * sz, -d / 2, d / 2),
  };
}

function resetPivot() {
  if (!selCount) return;
  let sx = 0, sz = 0;
  for (let i = 0; i < sel.length; i++) {
    if (!sel[i]) continue;
    const p = gridWorld(i);
    sx += p.x; sz += p.z;
  }
  const c = { x: sx / selCount, z: sz / selCount };
  Object.assign(pivot, state.snapPivot ? snapPivotXZ(c.x, c.z) : c);
}

/** Least-squares plane through the selected vertices (world units). */
function getFit() {
  if (fitCache) return fitCache;
  const s = state.heightScale;
  let n = 0, mx = 0, mz = 0, my = 0;
  const list = selectedList();
  const pts = list.map((i) => { const p = gridWorld(i); return { x: p.x, z: p.z, y: grid.h[i] * s }; });
  for (const p of pts) { mx += p.x; mz += p.z; my += p.y; n++; }
  if (!n) return (fitCache = { cx: 0, cz: 0, cy: 0, sx: 0, sz: 0 });
  mx /= n; mz /= n; my /= n;
  let sxx = 0, szz = 0, sxz = 0, sxy = 0, szy = 0;
  for (const p of pts) {
    const dx = p.x - mx, dz = p.z - mz, dy = p.y - my;
    sxx += dx * dx; szz += dz * dz; sxz += dx * dz; sxy += dx * dy; szy += dz * dy;
  }
  const det = sxx * szz - sxz * sxz;
  let gx = 0, gz = 0;
  const eps = 1e-9 * (sxx + szz + 1e-12);
  if (Math.abs(det) > eps * (sxx + szz)) {
    gx = (sxy * szz - szy * sxz) / det;
    gz = (szy * sxx - sxy * sxz) / det;
  } else if (sxx >= szz && sxx > 1e-12) gx = sxy / sxx;
  else if (szz > 1e-12) gz = szy / szz;
  return (fitCache = { cx: mx, cz: mz, cy: my, sx: gx, sz: gz });
}

function pivotHeight() {
  const f = getFit();
  return f.cy + f.sx * (pivot.x - f.cx) + f.sz * (pivot.z - f.cz);
}

/**
 * Apply per-vertex height offsets (world units) to the drag's vertices. If any
 * vertex would cross a limit, the whole move is scaled back so the selection
 * keeps its shape, and that vertex is flagged.
 */
function applyGroupDelta(g, deltaWorld) {
  const s = state.heightScale;
  let t = 1, lim = -1;
  for (let k = 0; k < g.list.length; k++) {
    const dn = deltaWorld[k] / s;
    const target = g.h0[k] + dn;
    let tk = 1;
    if (target > H_MAX + 1e-9) tk = (H_MAX - g.h0[k]) / dn;
    else if (target < H_MIN - 1e-9) tk = (H_MIN - g.h0[k]) / dn;
    tk = Math.max(0, tk);
    if (tk < t) { t = tk; lim = g.list[k]; }
  }
  for (let k = 0; k < g.list.length; k++) {
    grid.h[g.list[k]] = clamp(g.h0[k] + (t * deltaWorld[k]) / s, H_MIN, H_MAX);
  }
  if (lim !== limitIdx) { limitIdx = lim; scheduleSelection(); }
  scheduleGeometry();
}

// ---------------------------------------------------------------------------
// 3D pointer interaction
// ---------------------------------------------------------------------------

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const tmpPlane = new THREE.Plane();
const tmpVec = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);
const boxEl = $('box-select');

let drag = null;   // freeform vertex drag
let gdrag = null;  // pixel-mode gizmo drag
let box = null;    // pixel-mode box selection
let press = null;  // possible click (vs orbit)

function setRay(e) {
  const r = canvasEl.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
}

/** World position → client (screen) coordinates. */
function toScreen(x, y, z) {
  const r = canvasEl.getBoundingClientRect();
  tmpVec.set(x, y, z).project(camera);
  return { x: r.left + ((tmpVec.x + 1) / 2) * r.width, y: r.top + ((1 - tmpVec.y) / 2) * r.height, behind: tmpVec.z > 1 };
}

function claimPointer(e) {
  controls.enabled = false;
  canvasEl.setPointerCapture(e.pointerId);
  e.preventDefault();
  e.stopImmediatePropagation(); // keep OrbitControls from starting an orbit/pan
}

function releasePointer(e) {
  controls.enabled = true;
  if (canvasEl.hasPointerCapture(e.pointerId)) canvasEl.releasePointerCapture(e.pointerId);
}

function onPointerDown(e) {
  if (e.button !== 0) return;
  setRay(e);
  if (state.mode === 'pixel') return pixelPointerDown(e);

  const hit = raycaster.intersectObjects(handles.children, false)[0];
  if (hit) {
    const i = hit.object.userData.index;
    const v = state.verts[i];
    selectFree(i);
    let mode = state.dragMode;
    if (e.shiftKey) mode = mode === 'height' ? 'move' : 'height';
    if (v.corner) mode = 'height';
    const r = canvasEl.getBoundingClientRect();
    const dist = camera.position.distanceTo(hit.object.position);
    pushUndo();
    drag = {
      i, mode,
      startY: e.clientY,
      startH: v.h,
      unitsPerPx: (2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / r.height,
    };
    claimPointer(e);
    return;
  }
  press = { x: e.clientX, y: e.clientY };
}

function onPointerMove(e) {
  if (state.mode === 'pixel') return pixelPointerMove(e);
  if (!drag) return;
  const v = state.verts[drag.i];
  if (drag.mode === 'height') {
    const raw = drag.startH + ((drag.startY - e.clientY) * drag.unitsPerPx) / state.heightScale;
    v.h = Math.round(clamp(raw, H_MIN, H_MAX) * 1000) / 1000;
    const lim = raw > H_MAX || raw < H_MIN ? drag.i : -1;
    if (lim !== limitVertex) { limitVertex = lim; }
  } else {
    setRay(e);
    tmpPlane.set(UP, -v.h * state.heightScale);
    if (raycaster.ray.intersectPlane(tmpPlane, tmpVec)) {
      const { u, v: vv } = toUV(tmpVec.x, tmpVec.z);
      const nu = clamp(u, 0, 1), nv = clamp(vv, 0, 1);
      if (!tooClose(nu, nv, drag.i)) { v.u = nu; v.v = nv; }
    }
  }
  scheduleGeometry();
  updateFreeSelPanel();
}

function onPointerUp(e) {
  if (state.mode === 'pixel') return pixelPointerUp(e);
  if (drag) {
    drag = null;
    releasePointer(e);
    return;
  }
  if (press && e.type === 'pointerup' && e.button === 0) {
    const moved = Math.hypot(e.clientX - press.x, e.clientY - press.y);
    if (moved < 5) {
      setRay(e);
      const hit = raycaster.intersectObject(surface, false)[0];
      if (hit) addVertexAt(hit.point.x, hit.point.z, hit.point.y / state.heightScale);
      else selectFree(-1);
    }
  }
  press = null;
}

// --- Freeform helpers ---

function tooClose(u, v, ignore = -1) {
  const { w, d } = planeDims();
  return state.verts.some((p, i) => i !== ignore && Math.hypot((p.u - u) * w, (p.v - v) * d) < MIN_VERTEX_GAP * PLANE_LONG);
}

function addVertexAt(x, z, h) {
  const { u, v } = toUV(x, z);
  const nu = clamp(u, 0, 1), nv = clamp(v, 0, 1);
  if (tooClose(nu, nv)) return;
  pushUndo();
  state.verts.push({ u: nu, v: nv, h: Math.round(clamp(h, H_MIN, H_MAX) * 1000) / 1000 });
  selectFree(state.verts.length - 1);
  scheduleGeometry();
}

function deleteSelectedFree() {
  if (selected < 0) return;
  pushUndo();
  const v = state.verts[selected];
  if (v.corner) { v.h = 0; } // corners can't be removed; reset them instead
  else { state.verts.splice(selected, 1); selected = -1; }
  limitVertex = -1;
  updateFreeSelPanel();
  scheduleGeometry();
}

function selectFree(i) {
  selected = i;
  limitVertex = -1;
  updateFreeSelPanel();
  handles.children.forEach((m) => { m.material = handleMat(m.userData.index); });
}

// --- Pixel-mode pointer handling ---

function nearestGridPoint(clientX, clientY) {
  const s = state.heightScale;
  let best = -1, bestD = PICK_RADIUS_PX;
  for (let idx = 0; idx < grid.h.length; idx++) {
    const p = gridWorld(idx);
    const q = toScreen(p.x, grid.h[idx] * s, p.z);
    if (q.behind) continue;
    const dd = Math.hypot(q.x - clientX, q.y - clientY);
    if (dd < bestD) { bestD = dd; best = idx; }
  }
  return best;
}

/** Corner indices of the pixel under the cursor, or null. */
function pixelCornersAt(e) {
  setRay(e);
  const hit = raycaster.intersectObject(pSurface, false)[0];
  if (!hit) return null;
  const { u, v } = toUV(hit.point.x, hit.point.z);
  const i = clamp(Math.floor(u * grid.W), 0, grid.W - 1);
  const j = clamp(Math.floor(v * grid.H), 0, grid.H - 1);
  const W1 = grid.W + 1;
  return [j * W1 + i, j * W1 + i + 1, (j + 1) * W1 + i, (j + 1) * W1 + i + 1];
}

// Where grab zones overlap (e.g. a ring seen edge-on behind the arrows),
// prefer the smaller, more specific controls.
const PART_PRIORITY = { pivot: 0, arrow: 1, ringX: 2, ringZ: 2 };
function pickGizmo() {
  const hits = raycaster.intersectObjects(gizmo.pickables, false);
  let best = null;
  for (const h of hits) {
    if (!best || PART_PRIORITY[h.object.userData.part] < PART_PRIORITY[best.object.userData.part]) best = h;
  }
  return best;
}

function pixelPointerDown(e) {
  if (gizmo.group.visible) {
    const hit = pickGizmo();
    if (hit) {
      startGizmoDrag(hit.object.userData.part, hit.point, e);
      claimPointer(e);
      return;
    }
  }
  if (e.shiftKey || e.altKey) {
    box = { x0: e.clientX, y0: e.clientY, x1: e.clientX, y1: e.clientY, sub: e.altKey && !e.shiftKey };
    claimPointer(e);
    return;
  }
  press = { x: e.clientX, y: e.clientY };
}

function pixelPointerMove(e) {
  if (gdrag) { moveGizmoDrag(e); return; }
  if (box) {
    box.x1 = e.clientX; box.y1 = e.clientY;
    if (Math.hypot(box.x1 - box.x0, box.y1 - box.y0) >= 5) {
      const vr = viewport.getBoundingClientRect();
      boxEl.hidden = false;
      boxEl.classList.toggle('subtract', box.sub);
      Object.assign(boxEl.style, {
        left: `${Math.min(box.x0, box.x1) - vr.left}px`,
        top: `${Math.min(box.y0, box.y1) - vr.top}px`,
        width: `${Math.abs(box.x1 - box.x0)}px`,
        height: `${Math.abs(box.y1 - box.y0)}px`,
      });
    }
    return;
  }
  // Hover feedback on the gizmo
  if (gizmo.group.visible && e.buttons === 0) {
    setRay(e);
    const hit = pickGizmo();
    gizmo.setHover(hit ? hit.object.userData.part : null);
    canvasEl.style.cursor = hit ? (hit.object.userData.part === 'pivot' ? 'move' : 'grab') : '';
  }
}

function pixelPointerUp(e) {
  if (gdrag) {
    gdrag = null;
    gizmo.setHover(null);
    releasePointer(e);
    return;
  }
  if (box) {
    releasePointer(e);
    boxEl.hidden = true;
    const b = box;
    box = null;
    if (Math.hypot(b.x1 - b.x0, b.y1 - b.y0) < 5) { clickSelect(e, b.sub ? 'sub' : 'toggle'); return; }
    const x0 = Math.min(b.x0, b.x1), x1 = Math.max(b.x0, b.x1);
    const y0 = Math.min(b.y0, b.y1), y1 = Math.max(b.y0, b.y1);
    const s = state.heightScale;
    for (let idx = 0; idx < grid.h.length; idx++) {
      const p = gridWorld(idx);
      const q = toScreen(p.x, grid.h[idx] * s, p.z);
      if (!q.behind && q.x >= x0 && q.x <= x1 && q.y >= y0 && q.y <= y1) setSelected(idx, !b.sub);
    }
    selectionChanged();
    return;
  }
  if (press && e.type === 'pointerup' && e.button === 0) {
    if (Math.hypot(e.clientX - press.x, e.clientY - press.y) < 5) clickSelect(e, 'replace');
  }
  press = null;
}

function clickSelect(e, how) {
  const idx = nearestGridPoint(e.clientX, e.clientY);
  const targets = idx >= 0 ? [idx] : pixelCornersAt(e);
  if (how === 'replace') { sel.fill(0); selCount = 0; }
  if (targets) {
    if (how === 'sub') targets.forEach((i) => setSelected(i, false));
    else if (how === 'toggle') {
      const allOn = targets.every((i) => sel[i]);
      targets.forEach((i) => setSelected(i, !allOn));
    } else targets.forEach((i) => setSelected(i, true));
  }
  selectionChanged();
}

// --- Gizmo dragging ---

function startGizmoDrag(part, hitPoint, e) {
  const list = selectedList();
  const fit = getFit();
  const py = pivotHeight();
  const center = new THREE.Vector3(pivot.x, py, pivot.z);
  gdrag = {
    part, list,
    h0: Float32Array.from(list, (i) => grid.h[i]),
    xs: Float32Array.from(list, (i) => gridWorld(i).x),
    zs: Float32Array.from(list, (i) => gridWorld(i).z),
    sx0: fit.sx, sz0: fit.sz,
    pivotY: py,
    mx0: e.clientX, my0: e.clientY,
  };
  gizmo.setHover(part);
  limitIdx = -1;
  scheduleSelection();
  if (part === 'pivot') return;
  pushUndo();

  const scale = gizmo.group.scale.x;
  let axis = null, grab = null;
  if (part === 'arrow') {
    axis = UP.clone();
    grab = center.clone().addScaledVector(UP, 0.6 * scale);
  } else {
    axis = part === 'ringX' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
    // Nearest point on the ring to where it was grabbed.
    const off = hitPoint.clone().sub(center);
    off.addScaledVector(axis, -off.dot(axis));
    if (off.lengthSq() < 1e-12) off.set(0, 1, 0);
    grab = center.clone().add(off.setLength(RING_RADIUS * scale));
  }
  // Screen direction the grab point moves in for a small step of the control.
  const STEP = 0.05;
  let moved;
  if (part === 'arrow') moved = grab.clone().addScaledVector(UP, STEP * scale);
  else moved = grab.clone().sub(center).applyAxisAngle(axis, STEP).add(center);
  const a = toScreen(grab.x, grab.y, grab.z), b = toScreen(moved.x, moved.y, moved.z);
  const v = { x: (b.x - a.x) / STEP, y: (b.y - a.y) / STEP }; // pixels per unit of control
  const r = canvasEl.getBoundingClientRect();
  const dist = camera.position.distanceTo(center);
  const worldPerPx = (2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / r.height;
  gdrag.screenDir = v;
  // Seen edge-on, the control barely moves on screen; fall back to vertical mouse travel.
  gdrag.fallback = Math.hypot(v.x, v.y) < (part === 'arrow' ? 15 : 40);
  gdrag.worldPerPx = worldPerPx;
  gdrag.scale = scale;
}

function moveGizmoDrag(e) {
  const g = gdrag;
  if (g.part === 'pivot') {
    setRay(e);
    let x, z;
    const hit = raycaster.intersectObject(pSurface, false)[0];
    if (hit) { x = hit.point.x; z = hit.point.z; } else {
      tmpPlane.set(UP, -g.pivotY);
      if (!raycaster.ray.intersectPlane(tmpPlane, tmpVec)) return;
      x = tmpVec.x; z = tmpVec.z;
    }
    const { w, d } = planeDims();
    const p = state.snapPivot && !e.altKey ? snapPivotXZ(x, z) : { x: clamp(x, -w / 2, w / 2), z: clamp(z, -d / 2, d / 2) };
    Object.assign(pivot, p);
    return;
  }
  const dx = e.clientX - g.mx0, dy = e.clientY - g.my0;
  // Amount of control (radians for rings, world units for the arrow) from mouse travel.
  let amount;
  if (g.fallback) {
    amount = g.part === 'arrow' ? -dy * g.worldPerPx : -dy * 0.01;
  } else {
    const v = g.screenDir;
    amount = (dx * v.x + dy * v.y) / (v.x * v.x + v.y * v.y);
    if (g.part === 'arrow') amount *= g.scale;
  }
  const delta = new Float32Array(g.list.length);
  if (g.part === 'arrow') {
    delta.fill(amount);
  } else if (g.part === 'ringZ') {
    // Rotating about +Z tilts +X upward: the X slope's angle grows.
    const sxn = Math.tan(clamp(Math.atan(g.sx0) + amount, -MAX_TILT, MAX_TILT));
    for (let k = 0; k < delta.length; k++) delta[k] = (g.xs[k] - pivot.x) * (sxn - g.sx0);
  } else {
    // Rotating about +X tilts +Z downward: the Z slope's angle shrinks.
    const szn = Math.tan(clamp(Math.atan(g.sz0) - amount, -MAX_TILT, MAX_TILT));
    for (let k = 0; k < delta.length; k++) delta[k] = (g.zs[k] - pivot.z) * (szn - g.sz0);
  }
  applyGroupDelta(g, delta);
}

// --- Keyboard ---

window.addEventListener('keydown', (e) => {
  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
  if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); return; }
  if (state.mode === 'pixel') {
    if (mod && e.key.toLowerCase() === 'a') { e.preventDefault(); selectAll(); }
    else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); setSelectionHeight(0); }
    else if (e.key === 'Escape') clearSelection();
    return;
  }
  if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); deleteSelectedFree(); }
  if (e.key === 'Escape') selectFree(-1);
});

// ---------------------------------------------------------------------------
// Side panels
// ---------------------------------------------------------------------------

function fmt(n, digits = 3) { return String(Math.round(n * 10 ** digits) / 10 ** digits); }

function updateFreeSelPanel() {
  const v = state.mode === 'free' ? state.verts[selected] : null;
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
  const el = $(id);
  undoOnFirstInput(el);
  el.addEventListener('input', (e) => {
    const v = state.verts[selected];
    const n = parseFloat(e.target.value);
    if (!v || !Number.isFinite(n)) return;
    apply(v, n);
    scheduleGeometry();
  });
  el.addEventListener('change', updateFreeSelPanel);
}
bindSelField('sel-x', (v, n) => { const u = clamp(n / state.texW, 0, 1); if (!tooClose(u, v.v, selected)) v.u = u; });
bindSelField('sel-y', (v, n) => { const vv = clamp(n / state.texH, 0, 1); if (!tooClose(v.u, vv, selected)) v.v = vv; });
bindSelField('sel-h', (v, n) => { v.h = clamp(n, H_MIN, H_MAX); });
$('btn-delete').addEventListener('click', deleteSelectedFree);

function updatePixelPanel() {
  if (!grid) return;
  $('pix-sel-count').textContent = selCount
    ? `${selCount} of ${sel.length} vertices selected`
    : 'No vertices selected';
  $('btn-set-height').disabled = !selCount;
}

function selectAll() { sel.fill(1); selCount = sel.length; selectionChanged(); }
function clearSelection() { sel.fill(0); selCount = 0; selectionChanged(); }
function setSelectionHeight(hv) {
  if (!selCount) return;
  pushUndo();
  const v = clamp(hv, H_MIN, H_MAX);
  for (let i = 0; i < sel.length; i++) if (sel[i]) grid.h[i] = v;
  limitIdx = -1;
  fitCache = null;
  scheduleGeometry();
  scheduleSelection();
}
$('btn-select-all').addEventListener('click', selectAll);
$('btn-clear-sel').addEventListener('click', clearSelection);
$('btn-set-height').addEventListener('click', () => {
  const n = parseFloat($('set-height').value);
  if (Number.isFinite(n)) setSelectionHeight(n);
});
$('snap-pivot').addEventListener('change', (e) => {
  state.snapPivot = e.target.checked;
  if (state.snapPivot && grid) Object.assign(pivot, snapPivotXZ(pivot.x, pivot.z));
  saveState();
});
$('btn-from-free').addEventListener('click', () => {
  if (!confirm('Replace the pixel grid with heights sampled from the freeform surface?')) return;
  pushUndo();
  grid = gridFromFreeform(state.texW, state.texH);
  resetSelection();
  scheduleGeometry();
});

function updateStats() {
  $('stats').textContent = state.mode === 'pixel'
    ? `${grid.W}×${grid.H} pixels · ${grid.h.length} vertices · ${grid.W * grid.H * 4} triangles`
    : `${state.verts.length} vertices · ${mesh.triangles.length / 3} triangles`;
}

// ---------------------------------------------------------------------------
// Surface mode
// ---------------------------------------------------------------------------

function setMode(mode) {
  if (mode === 'pixel' && Math.max(state.texW, state.texH) > PIXEL_MAX) {
    updateModeUI(`Pixel-perfect mode supports textures up to ${PIXEL_MAX}×${PIXEL_MAX}. Reduce the size first.`);
    return;
  }
  if (mode === state.mode) return;
  if (mode === 'pixel') {
    if (!grid || grid.W !== state.texW || grid.H !== state.texH) {
      grid = grid ? resampleGrid(grid, state.texW, state.texH) : gridFromFreeform(state.texW, state.texH);
    }
    resetSelection();
  }
  state.mode = mode;
  selected = -1;
  limitVertex = -1;
  syncAllUI();
  scheduleGeometry();
}

function updateModeUI(message) {
  const pixel = state.mode === 'pixel';
  document.querySelectorAll('#surface-mode button').forEach((b) => b.classList.toggle('active', b.dataset.v === state.mode));
  $('mode-hint').textContent = message || (pixel
    ? 'One vertex at every pixel corner, four triangles per pixel, one normal per pixel.'
    : 'Place vertices anywhere; the surface is triangulated between them.');
  $('free-edit').hidden = pixel;
  $('free-sel').hidden = pixel;
  $('pixel-edit').hidden = !pixel;
  $('help-free').hidden = pixel;
  $('help-pixel').hidden = !pixel;
  freeGroup.visible = !pixel;
  pixelGroup.visible = pixel;
  $('tex-w').max = $('tex-h').max = pixel ? PIXEL_MAX : TEX_MAX;
  $('snap-pivot').checked = state.snapPivot;
  updateToolUI();
}
document.querySelectorAll('#surface-mode button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.v)));

// ---------------------------------------------------------------------------
// Texture settings & export
// ---------------------------------------------------------------------------

function setTexSize(w, h) {
  const max = state.mode === 'pixel' ? PIXEL_MAX : TEX_MAX;
  const W = clamp(Math.round(w) || 1, 1, max);
  const H = clamp(Math.round(h) || 1, 1, max);
  $('tex-w').value = W;
  $('tex-h').value = H;
  if (W === state.texW && H === state.texH) return;
  pushUndo();
  state.texW = W;
  state.texH = H;
  // Larger textures leave the grid as it was; it's resampled on returning to pixel mode.
  if (grid && Math.max(W, H) <= PIXEL_MAX) {
    grid = resampleGrid(grid, W, H);
    resetSelection();
  }
  resampleCurvature(W, H);
  saveCurvature();
  // Keep the current viewing angle, but move in/out so the resized plane fits.
  const offset = camera.position.clone().sub(controls.target);
  camera.position.copy(controls.target).add(offset.setLength(fitDistance()));
  scheduleGeometry();
  updateFreeSelPanel();
}
$('tex-w').addEventListener('change', (e) => setTexSize(+e.target.value, state.texH));
$('tex-h').addEventListener('change', (e) => setTexSize(state.texW, +e.target.value));
$('btn-swap').addEventListener('click', () => setTexSize(state.texH, state.texW));

$('convention').addEventListener('change', (e) => {
  state.convention = e.target.value;
  if (state.previewKind === 'normal') scheduleTexture(); else saveState();
});

const heightScaleEl = $('height-scale');
undoOnFirstInput(heightScaleEl);
heightScaleEl.addEventListener('input', (e) => {
  const n = parseFloat(e.target.value);
  if (!Number.isFinite(n) || n <= 0) return;
  state.heightScale = clamp(n, 0.01, 100);
  fitCache = null;
  scheduleGeometry();
});
heightScaleEl.addEventListener('change', () => { heightScaleEl.value = state.heightScale; });

function canvasPng(c, name) {
  c.toBlob((blob) => download(blob, name), 'image/png');
}

function exportPng(kind) {
  const W = state.texW, H = state.texH;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(W, H);
  if (state.mode === 'pixel') {
    computePixelSurface();
    if (kind === 'height') encodeHeight(pixelSurf.hc, W * H, img.data);
    else encodeNormalArray(pixelSurf.n, W * H, img.data);
  } else {
    const r = rasterize(W, H, kind === 'height');
    if (kind === 'height') {
      const s = state.heightScale;
      encodeHeight(r.height.map((x) => x * s), W * H, img.data);
    } else encodeNormalTri(r, img.data);
  }
  ctx.putImageData(img, 0, 0);
  canvasPng(c, `${kind === 'height' ? 'height' : `normal_${state.convention}`}_${W}x${H}.png`);
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
  const project = { ...serialize(), curvature: curvDataUrl() };
  download(new Blob([JSON.stringify(project)], { type: 'application/json' }), 'normal-map-project.json');
});
$('btn-load-json').addEventListener('click', () => $('file-json').click());
$('file-json').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    const s = JSON.parse(await file.text());
    if (!isValidState(s)) throw new Error('not a Normal Map Builder project');
    pushUndo();
    await applyState(s, s.curvature || null);
  } catch (err) {
    alert(`Could not load that file: ${err.message}`);
  }
});
$('btn-reset').addEventListener('click', () => {
  if (!confirm('Discard all vertices, the pixel grid, the curvature map and settings?')) return;
  pushUndo();
  applyState(defaultState(), null);
});

// ---------------------------------------------------------------------------
// 2D texture preview tools: pixel selection & curvature painting
// ---------------------------------------------------------------------------

let texDrag = null;

function texPixelAt(e) {
  const r = normalCanvas.getBoundingClientRect();
  return {
    x: clamp(Math.floor(((e.clientX - r.left) / r.width) * state.texW), 0, state.texW - 1),
    y: clamp(Math.floor(((e.clientY - r.top) / r.height) * state.texH), 0, state.texH - 1),
  };
}

function paintAt(x, y, value) {
  const n = state.brush.size;
  const x0 = x - Math.floor((n - 1) / 2), y0 = y - Math.floor((n - 1) / 2);
  for (let j = Math.max(0, y0); j < Math.min(curv.H, y0 + n); j++) {
    for (let i = Math.max(0, x0); i < Math.min(curv.W, x0 + n); i++) curv.data[j * curv.W + i] = value;
  }
}

function paintLine(a, b, value) {
  const steps = Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y), 1);
  for (let s = 0; s <= steps; s++) {
    paintAt(Math.round(a.x + ((b.x - a.x) * s) / steps), Math.round(a.y + ((b.y - a.y) * s) / steps), value);
  }
}

texFrame.addEventListener('contextmenu', (e) => e.preventDefault());
texFrame.addEventListener('pointerdown', (e) => {
  const tool = activeTool();
  if (!tool || (e.button !== 0 && e.button !== 2)) return;
  if (e.button === 2 && tool !== 'paint') return;
  texFrame.setPointerCapture(e.pointerId);
  const p = texPixelAt(e);
  if (tool === 'paint') {
    pushUndo();
    const value = e.button === 2 ? 0 : state.brush.value;
    texDrag = { tool, last: p, value };
    paintAt(p.x, p.y, value);
    scheduleCurvature();
  } else {
    const how = e.altKey && !e.shiftKey ? 'sub' : e.shiftKey ? 'add' : 'replace';
    texDrag = { tool, how };
    texDragRect = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, sub: how === 'sub' };
    drawOverlay();
  }
});
texFrame.addEventListener('pointermove', (e) => {
  if (!texDrag) return;
  const p = texPixelAt(e);
  if (texDrag.tool === 'paint') {
    paintLine(texDrag.last, p, texDrag.value);
    texDrag.last = p;
    scheduleCurvature();
  } else {
    texDragRect.x1 = p.x; texDragRect.y1 = p.y;
    drawOverlay();
  }
});
function endTexDrag() {
  if (!texDrag) return;
  if (texDrag.tool === 'select' && texDragRect) {
    const { x0, y0, x1, y1 } = texDragRect;
    if (texDrag.how === 'replace') { sel.fill(0); selCount = 0; }
    const W1 = grid.W + 1;
    for (let j = Math.min(y0, y1); j <= Math.max(y0, y1) + 1; j++) {
      for (let i = Math.min(x0, x1); i <= Math.max(x0, x1) + 1; i++) setSelected(j * W1 + i, texDrag.how !== 'sub');
    }
    texDragRect = null;
    selectionChanged();
  }
  texDrag = null;
}
texFrame.addEventListener('pointerup', endTexDrag);
texFrame.addEventListener('pointercancel', endTexDrag);

function activeTool() {
  if (state.tool2d === 'select') return state.mode === 'pixel' ? 'select' : null;
  return 'paint';
}

function updateToolUI() {
  const pixel = state.mode === 'pixel';
  document.querySelectorAll('#tool2d button').forEach((b) => {
    b.classList.toggle('active', b.dataset.v === state.tool2d);
    if (b.dataset.v === 'select') b.disabled = !pixel;
  });
  const tool = activeTool();
  texFrame.classList.toggle('tool-select', tool === 'select');
  texFrame.classList.toggle('tool-paint', tool === 'paint');
  $('paint-controls').hidden = state.tool2d !== 'paint';
  $('tool-hint').textContent = state.tool2d === 'paint'
    ? 'Left-drag paints the curvature value; right-drag erases to black. Lighter pixels shade as if rounded.'
    : pixel
      ? 'Drag across pixels to select their corners. Shift adds, Alt removes.'
      : 'Pixel selection works in pixel-perfect mode. Switch to Paint curvature to edit the curvature map.';
  $('brush-size').value = state.brush.size;
  $('brush-value').value = state.brush.value;
  $('brush-value-out').textContent = `${Math.round((state.brush.value / 255) * 100)}%`;
}

document.querySelectorAll('#tool2d button').forEach((b) => b.addEventListener('click', () => {
  state.tool2d = b.dataset.v;
  if (state.tool2d === 'paint' && state.previewKind !== 'curvature') {
    state.previewKind = 'curvature';
    syncSegs();
    drawTexturePreview();
  }
  updateToolUI();
  saveState();
}));
$('brush-size').addEventListener('change', (e) => {
  state.brush.size = clamp(Math.round(+e.target.value) || 1, 1, 64);
  updateToolUI();
  saveState();
});
$('brush-value').addEventListener('input', (e) => {
  state.brush.value = +e.target.value;
  updateToolUI();
  saveState();
});
$('btn-curv-clear').addEventListener('click', () => {
  if (curvIsEmpty()) return;
  pushUndo();
  curv.data.fill(0);
  scheduleCurvature();
});
$('btn-curv-export').addEventListener('click', () => canvasPng(curvatureCanvas(), `curvature_${curv.W}x${curv.H}.png`));
$('btn-curv-import').addEventListener('click', () => $('file-curv').click());
$('file-curv').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    pushUndo();
    await loadCurvatureImage(url);
    scheduleCurvature();
  } catch (err) {
    alert(`Could not load that image: ${err.message}`);
  } finally {
    URL.revokeObjectURL(url);
  }
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
  $('curve-model').value = state.light.curveModel;
}
$('light-type').addEventListener('change', (e) => { state.light.type = e.target.value; scheduleLit(); saveState(); });
$('light-height').addEventListener('input', (e) => { state.light.height = +e.target.value; syncLightControls(); scheduleLit(); saveState(); });
$('ambient').addEventListener('input', (e) => { state.light.ambient = +e.target.value; syncLightControls(); scheduleLit(); saveState(); });
$('curve-model').addEventListener('change', (e) => { state.light.curveModel = e.target.value; scheduleLit(); saveState(); });
$('btn-export-lut').addEventListener('click', () => {
  const c = document.createElement('canvas');
  c.width = LUT_W; c.height = LUT_H;
  c.getContext('2d').putImageData(new ImageData(lutImagePixels(coneLut), LUT_W, LUT_H), 0, 0);
  canvasPng(c, `curvature_cone_lut_${LUT_W}x${LUT_H}.png`);
});

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
  renderGradientUI();
  const marker = gradMarkers.children[i];
  marker.setPointerCapture(e.pointerId);
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
// Misc UI wiring & startup
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

function syncAllUI() {
  $('tex-w').value = state.texW;
  $('tex-h').value = state.texH;
  $('convention').value = state.convention;
  heightScaleEl.value = state.heightScale;
  syncSegs();
  syncLightControls();
  renderGradientUI();
  updateModeUI();
  updateFreeSelPanel();
  updatePixelPanel();
  updateUndoButtons();
}

/** Replace the whole document state (startup, load, reset). */
async function applyState(raw, curvatureUrl) {
  state = normalizeState(raw);
  grid = gridFromSerialized(raw.grid);
  if (state.mode === 'pixel') {
    if (!grid) grid = gridFromFreeform(state.texW, state.texH);
    else if (grid.W !== state.texW || grid.H !== state.texH) grid = resampleGrid(grid, state.texW, state.texH);
  }
  curv = { W: state.texW, H: state.texH, data: new Uint8Array(state.texW * state.texH) };
  if (curvatureUrl) {
    try { await loadCurvatureImage(curvatureUrl); } catch { /* keep the blank map */ }
  }
  selected = -1;
  limitVertex = -1;
  selectedStop = 0;
  resetSelection();
  syncAllUI();
  scheduleGeometry();
  saveCurvature();
}

let storedCurvature = null;
try { storedCurvature = localStorage.getItem(CURV_KEY); } catch { /* storage unavailable */ }
applyState(loadState() ?? defaultState(), storedCurvature);

// Test hook: open with ?debug to inspect internals from the console.
if (new URLSearchParams(location.search).has('debug')) {
  window.__nmb = {
    get state() { return state; }, get grid() { return grid; }, get sel() { return sel; },
    get limitIdx() { return limitIdx; }, get curv() { return curv; },
    camera, gizmo, pivot, toScreen, pivotHeight,
  };
}
