import type { SceneId } from "../story/pages";
import { fall, rabbitScene, riverbank, waking } from "./above";
import type { SceneBuilder } from "./kit";
import { caterpillarScene, cheshire, croquet, drinkMe, eatMe, teaParty } from "./wonderland";

export const SCENES: Record<SceneId, SceneBuilder> = {
  riverbank,
  rabbit: rabbitScene,
  fall,
  drinkMe,
  eatMe,
  caterpillar: caterpillarScene,
  cheshire,
  teaParty,
  croquet,
  waking,
};
