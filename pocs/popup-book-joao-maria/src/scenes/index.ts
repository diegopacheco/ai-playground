import type { SceneId } from "../story/pages";
import { ovenScene, theEnd } from "./finale";
import type { SceneBuilder } from "./kit";
import { cover, crumbs, lost, woodcutter } from "./opening";
import { boneScene, cageScene, candyHouseScene, witchScene } from "./witch";

export const SCENES: Record<SceneId, SceneBuilder> = {
  cover,
  woodcutter,
  crumbs,
  lost,
  candyHouse: candyHouseScene,
  witch: witchScene,
  cage: cageScene,
  bone: boneScene,
  oven: ovenScene,
  theEnd,
};
