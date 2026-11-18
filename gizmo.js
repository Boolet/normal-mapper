import * as THREE from 'three';

// Transform tool for pixel-perfect mode: a pivot dot, a ring around the X axis,
// a ring around the Z axis, and an up/down arrow. Built at unit size; the caller
// positions and scales the group every frame.

const COLORS = { pivot: 0xffffff, ringX: 0xff5d5d, ringZ: 0x5d8bff, arrow: 0x5ee08a };
const HOVER = 0xffe066;
export const RING_RADIUS = 1;

export function createGizmo() {
  const group = new THREE.Group();
  const mats = {};
  const pickables = [];
  const pickMat = new THREE.MeshBasicMaterial({ visible: false });

  function part(name, visualGeos, pickGeos) {
    const g = new THREE.Group();
    const m = new THREE.MeshBasicMaterial({ color: COLORS[name], depthTest: false, depthWrite: false, transparent: true });
    mats[name] = m;
    for (const geo of visualGeos) {
      const mesh = new THREE.Mesh(geo, m);
      mesh.renderOrder = 20;
      g.add(mesh);
    }
    for (const geo of pickGeos) {
      const p = new THREE.Mesh(geo, pickMat);
      p.userData.part = name;
      g.add(p);
      pickables.push(p);
    }
    group.add(g);
    return g;
  }

  const ringGeo = new THREE.TorusGeometry(RING_RADIUS, 0.028, 8, 96);
  const ringPick = new THREE.TorusGeometry(RING_RADIUS, 0.13, 8, 48);
  // TorusGeometry lies in the XY plane, so its axis is Z.
  part('ringZ', [ringGeo], [ringPick]);
  part('ringX', [ringGeo], [ringPick]).rotation.y = Math.PI / 2;

  const shaftLen = 0.6, shaftStart = 0.22, headLen = 0.26;
  const shaft = new THREE.CylinderGeometry(0.032, 0.032, shaftLen, 10).translate(0, shaftStart + shaftLen / 2, 0);
  const head = new THREE.ConeGeometry(0.1, headLen, 18).translate(0, shaftStart + shaftLen + headLen / 2, 0);
  const pickUp = new THREE.CylinderGeometry(0.16, 0.16, shaftLen + headLen, 8).translate(0, shaftStart + (shaftLen + headLen) / 2, 0);
  part('arrow',
    [shaft, head, shaft.clone().rotateX(Math.PI), head.clone().rotateX(Math.PI)],
    [pickUp, pickUp.clone().rotateX(Math.PI)]);

  part('pivot', [new THREE.SphereGeometry(0.1, 20, 14)], [new THREE.SphereGeometry(0.2, 12, 8)]);

  let hovered = null;
  function setHover(name) {
    if (name === hovered) return;
    if (hovered) mats[hovered].color.set(COLORS[hovered]);
    hovered = name;
    if (hovered) mats[hovered].color.set(HOVER);
  }

  return { group, pickables, setHover };
}
