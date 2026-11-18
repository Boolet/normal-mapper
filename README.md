# Normal Map Builder

A browser tool for building normal maps by sculpting a low-poly heightfield.

## Surface modes

**Freeform.** Click the plane to add a vertex anywhere. Drag a vertex to change its height, or Shift-drag (or switch the mode) to slide it across the plane. The surface between vertices is Delaunay-triangulated, so each facet becomes one flat normal.

**Pixel-perfect** (textures up to 256×256). There is a vertex at every pixel corner, and each pixel is drawn as four triangles meeting at its center. The pixel's normal is the average plane of its four corners, so every pixel gets exactly one normal.
- **Select:** click a vertex or pixel. Shift-click toggles, Shift-drag box-selects, Alt-drag deselects. You can also drag across pixels in the texture preview. Ctrl+A selects all and Esc clears.
- **Transform tool:**
  - Drag the white **pivot** to move it. It snaps to pixel corners and centers; hold Alt to skip snapping.
  - Drag the red **X ring** or blue **Z ring** to tilt the selection around the pivot.
  - Drag the green **arrows** to raise or lower the whole selection.

  Vertices only ever move up and down. A tilt changes the slope of the selection's best-fit plane, and each vertex moves in proportion to its distance from the pivot, so a flat selection stays flat.
- Del flattens the selection to 0. **Set height** sets every selected vertex to one value.

**Height limits.** Vertex heights are limited to −1 … 1. If a tilt or raise would push any vertex past a limit, the whole selection stops together, so it doesn't deform, and the vertex that hit the limit turns red. **Height scale** multiplies all heights, flattening or exaggerating the result without re-editing. Plane units: the long side is 10.

## Previews and output

- **Texture preview:** shows the normal map (OpenGL Y+ or DirectX Y−), the heightmap, or the curvature map. Exports PNGs at full resolution.
  - The heightmap maps −1 … 1 to black … white.
- **Lighting preview:** drag a point or directional light. The light level is mapped through a color gradient, or banded into steps with a color for each.
- **Undo/redo** (Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y) covers the surface, texture size, height scale and curvature map.
- Work autosaves to the browser. Save project / Load project round-trips everything as JSON, including the curvature map.

## Curvature map

A grayscale texture, painted in the preview or imported as a PNG. It makes a low-res step read as a curve, using only the pixel being shaded and no neighbors. A value of c lets that pixel's normal behave as if it could turn by up to c × 45°. It only affects the lighting preview and is exported separately. To get the same look in a game, store it in a spare channel (e.g. the normal map's alpha) and use one of these models:

**Bend toward light.** The normal turns toward the light by up to the allowed angle. Edges catch light; shading never gets darker.

```hlsl
float BendDiffuse(float NdotL, float curvature)
{
    float phi = acos(clamp(NdotL, -1.0, 1.0));
    float d = max(phi - curvature * 0.7853982, 0.0); // 45° in radians
    return d >= 1.5707963 ? 0.0 : cos(d);
}
```

**Cone average.** Diffuse light is averaged over every normal in the cone, as if a rounded bevel were squeezed into one pixel. Lit sides soften and shadowed sides pick up some light. It's normalized so a pixel facing the light stays at full brightness. This one is precomputed: click **Export cone LUT** and sample the 128×64 table. Use bilinear filtering, clamp addressing and no mipmaps.

```hlsl
float ConeDiffuse(float NdotL, float curvature)
{
    return SAMPLE_TEXTURE2D(_ConeLut, sampler_ConeLut, float2(NdotL * 0.5 + 0.5, curvature)).r;
}
```

The LUT's curvature-0 row is at the bottom of the image, matching bottom-left UV origins (Unity, OpenGL). For top-left origins, flip the v coordinate.

## Running

It's a static site with no build step. Three.js and Delaunator load from jsDelivr.

Locally:

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000. ES modules don't load from `file://`, so opening `index.html` directly won't work. Add `?debug` to the URL to expose internals as `window.__nmb`.

On GitHub Pages: Settings → Pages → Deploy from a branch → `main` / `(root)`.

## Conventions

- Image X = +X in the 3D view. Image down = +Z, so the top of the texture is the far edge in the default view.
- OpenGL: green = surface faces up the image. DirectX flips green.
