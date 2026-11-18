// Curvature-aware diffuse shading.
//
// A curvature value c in [0, 1] lets a pixel's normal act as if it could turn
// by up to c × 45° (MAX_BEND). Two models are offered:
//
//  • bend: the normal turns toward the light by up to that angle, i.e. the
//    pixel shows the brightest normal in its cone. Only ever brightens.
//  • cone: the diffuse term is averaged over every normal in the cone, like a
//    rounded surface squeezed into one pixel. Lit sides soften and shadowed
//    sides pick up light. Normalized so that a pixel facing the light stays at 1.

export const MAX_BEND = Math.PI / 4;

// Cone lookup table: columns = N·L from -1 to 1, rows = curvature from 0 to 1.
export const LUT_W = 128;
export const LUT_H = 64;

/** Diffuse term for the "bend toward light" model. */
export function bendDiffuse(ndotl, c) {
  const phi = Math.acos(ndotl > 1 ? 1 : ndotl < -1 ? -1 : ndotl);
  const d = phi - c * MAX_BEND;
  if (d <= 0) return 1;
  if (d >= Math.PI / 2) return 0;
  return Math.cos(d);
}

/** Precompute the cone-average table by numerically integrating over each cone. */
export function buildConeLut() {
  const lut = new Float32Array(LUT_W * LUT_H);
  const RINGS = 16, AZ = 32;
  for (let r = 0; r < LUT_H; r++) {
    const a = (r / (LUT_H - 1)) * MAX_BEND;
    // Sample directions inside the cone (uniform in solid angle via sinθ weights).
    const samples = [];
    if (a > 1e-6) {
      for (let i = 0; i < RINGS; i++) {
        const th = ((i + 0.5) / RINGS) * a;
        const w = Math.sin(th);
        for (let j = 0; j < AZ; j++) {
          const ps = ((j + 0.5) / AZ) * Math.PI * 2;
          samples.push(Math.cos(th), Math.sin(th) * Math.cos(ps), w);
        }
      }
    }
    const avg = (cosPhi) => {
      if (!samples.length) return Math.max(0, cosPhi);
      const sinPhi = Math.sqrt(Math.max(0, 1 - cosPhi * cosPhi));
      let sum = 0, wsum = 0;
      for (let s = 0; s < samples.length; s += 3) {
        const d = samples[s] * cosPhi + samples[s + 1] * sinPhi;
        if (d > 0) sum += samples[s + 2] * d;
        wsum += samples[s + 2];
      }
      return sum / wsum;
    };
    const norm = avg(1);
    for (let x = 0; x < LUT_W; x++) {
      lut[r * LUT_W + x] = Math.min(1, avg((x / (LUT_W - 1)) * 2 - 1) / norm);
    }
  }
  return lut;
}

/** Bilinear lookup into the cone table. */
export function sampleCone(lut, ndotl, c) {
  const fx = Math.min(LUT_W - 1, Math.max(0, (ndotl * 0.5 + 0.5) * (LUT_W - 1)));
  const fy = Math.min(LUT_H - 1, Math.max(0, c * (LUT_H - 1)));
  const x0 = fx | 0, y0 = fy | 0;
  const x1 = Math.min(LUT_W - 1, x0 + 1), y1 = Math.min(LUT_H - 1, y0 + 1);
  const tx = fx - x0, ty = fy - y0;
  const a = lut[y0 * LUT_W + x0] + (lut[y0 * LUT_W + x1] - lut[y0 * LUT_W + x0]) * tx;
  const b = lut[y1 * LUT_W + x0] + (lut[y1 * LUT_W + x1] - lut[y1 * LUT_W + x0]) * tx;
  return a + (b - a) * ty;
}

/**
 * RGBA pixels of the cone table for export. Curvature 0 is the bottom row, so
 * in engines with bottom-left UV origins (Unity, OpenGL) you sample it at
 * uv = (N·L * 0.5 + 0.5, curvature).
 */
export function lutImagePixels(lut) {
  const px = new Uint8ClampedArray(LUT_W * LUT_H * 4);
  for (let r = 0; r < LUT_H; r++) {
    const row = LUT_H - 1 - r;
    for (let x = 0; x < LUT_W; x++) {
      const g = Math.round(lut[r * LUT_W + x] * 255);
      const k = (row * LUT_W + x) * 4;
      px[k] = px[k + 1] = px[k + 2] = g;
      px[k + 3] = 255;
    }
  }
  return px;
}
