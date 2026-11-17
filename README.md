# Normal Map Builder

A browser tool for building normal maps by sculpting a low-poly heightfield.

- **3D editor:** a plane on the XZ axis whose proportions match the output texture. Click the surface to add a vertex. Drag a vertex to change its height (Y), or Shift-drag (or switch the mode) to slide it across the plane. The surface between vertices is Delaunay-triangulated, so each facet becomes one flat normal.
- **Texture preview:** shows the normal map (OpenGL Y+ or DirectX Y−) or the heightmap, and exports either one as a PNG at the chosen width and height (up to 4096).
- **Lighting preview:** drag a point or directional light around. The light level is mapped through a color gradient, or banded into equal steps with a color for each.
- Work is autosaved to the browser's local storage. **Save project / Load project** round-trips everything as JSON.

## Running

It's a static site with no build step. Three.js and Delaunator load from jsDelivr.

Local:

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000. ES modules don't load from `file://`, so opening `index.html` directly won't work.

GitHub Pages: put these files at the root of a repo, then go to **Settings → Pages → Build and deployment**, choose **Deploy from a branch**, and pick `main` / `(root)`.

## Conventions

- Image X = +X in the 3D view. Image down = +Z, so the top of the texture is the far edge in the default 3D view.
- OpenGL: green = surface faces up the image. DirectX flips green.
- Heights use the plane's units: the longer side of the plane is 10. Normals come from the true 3D slopes you see in the editor.
