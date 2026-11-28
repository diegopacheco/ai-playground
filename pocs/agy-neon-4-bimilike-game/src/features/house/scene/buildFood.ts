import * as THREE from "three";
import { getDishSpec } from "../../kitchen/recipes";
import type { CookedDish } from "../../kitchen/types";
import { clayMaterial, glossyMaterial } from "./materials";

export function buildFoodMesh(cooked: CookedDish): THREE.Group {
  const spec = getDishSpec(cooked.dishId);
  const group = new THREE.Group();
  const plateMat = glossyMaterial("#ffffff");
  const mainMat = clayMaterial(spec.mainColor, 0.55);
  const topMat = clayMaterial(spec.toppingColor, 0.48);

  const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.2, 0.04, 18), plateMat);
  plate.position.y = 0.02;
  plate.castShadow = true;
  group.add(plate);

  switch (cooked.dishId) {
    case "cupcake": {
      const base = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.1, 0.16, 14), mainMat);
      base.position.y = 0.11;
      group.add(base);
      const icing = new THREE.Mesh(new THREE.SphereGeometry(0.14, 14, 12), topMat);
      icing.position.y = 0.22;
      group.add(icing);
      const cherry = new THREE.Mesh(new THREE.SphereGeometry(0.045, 10, 8), glossyMaterial("#ff2a55"));
      cherry.position.y = 0.35;
      group.add(cherry);
      break;
    }
    case "pancakes": {
      for (let i = 0; i < 3; i += 1) {
        const cake = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.05, 16), mainMat);
        cake.position.y = 0.065 + i * 0.055;
        group.add(cake);
      }
      const butter = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.04, 0.08), topMat);
      butter.position.y = 0.23;
      group.add(butter);
      break;
    }
    case "smoothie": {
      const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.09, 0.26, 14), mainMat);
      cup.position.y = 0.16;
      group.add(cup);
      const cream = new THREE.Mesh(new THREE.SphereGeometry(0.11, 12, 10), topMat);
      cream.position.y = 0.28;
      group.add(cream);
      const straw = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.22, 8), glossyMaterial("#74c0fc"));
      straw.position.set(0.04, 0.38, 0);
      straw.rotation.z = -0.25;
      group.add(straw);
      break;
    }
    case "pizza": {
      const crust = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.045, 18), mainMat);
      crust.position.y = 0.06;
      group.add(crust);
      for (const [px, pz] of [[-0.07, -0.05], [0.08, -0.04], [0, 0.08]] as const) {
        const pep = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.02, 12), topMat);
        pep.position.set(px, 0.09, pz);
        group.add(pep);
      }
      break;
    }
    case "soup": {
      const bowl = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 12, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), plateMat);
      bowl.position.y = 0.17;
      group.add(bowl);
      const broth = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.03, 16), mainMat);
      broth.position.y = 0.16;
      group.add(broth);
      const herb = new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 8), topMat);
      herb.position.y = 0.18;
      group.add(herb);
      break;
    }
    case "honeyCake": {
      const cake = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.18, 16), mainMat);
      cake.position.y = 0.13;
      group.add(cake);
      const glaze = new THREE.Mesh(new THREE.CylinderGeometry(0.185, 0.185, 0.05, 16), topMat);
      glaze.position.y = 0.22;
      group.add(glaze);
      break;
    }
    case "cookie": {
      const cookie = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.05, 16), mainMat);
      cookie.position.y = 0.065;
      group.add(cookie);
      for (const [cx, cz] of [[-0.06, -0.04], [0.06, 0.05], [-0.03, 0.07], [0.07, -0.06]] as const) {
        const chip = new THREE.Mesh(new THREE.SphereGeometry(0.028, 8, 8), topMat);
        chip.position.set(cx, 0.095, cz);
        group.add(chip);
      }
      break;
    }
    default: {
      const unreachable: never = cooked.dishId;
      return unreachable;
    }
  }

  return group;
}
