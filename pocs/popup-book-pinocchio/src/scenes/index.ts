import type { SceneId } from "../story/pages";
import { fairy, field, foxCat, puppetShow } from "./journey";
import type { SceneBuilder } from "./kit";
import { alive, cover, cricketScene, workshop } from "./opening";
import { inside, sharkScene, theEnd, toyland } from "./sea";

export const SCENES: Record<SceneId, SceneBuilder> = {
  cover,
  workshop,
  alive,
  cricket: cricketScene,
  puppetShow,
  foxCat,
  field,
  fairy,
  toyland,
  shark: sharkScene,
  inside,
  theEnd,
};
