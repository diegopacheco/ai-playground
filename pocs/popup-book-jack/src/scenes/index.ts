import type { SceneId } from "../story/pages";
import { beansScene, beanstalkScene, cottageScene, cover, windowScene } from "./ground";
import type { SceneBuilder } from "./kit";
import { castleScene, chaseScene, cloudsScene, theEnd, treasureScene } from "./sky";

export const SCENES: Record<SceneId, SceneBuilder> = {
  cover,
  cottage: cottageScene,
  beans: beansScene,
  window: windowScene,
  beanstalk: beanstalkScene,
  clouds: cloudsScene,
  castle: castleScene,
  treasure: treasureScene,
  chase: chaseScene,
  theEnd,
};
