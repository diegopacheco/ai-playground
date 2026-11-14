import type { SceneId } from "../story/pages";
import { kassim, treasure } from "./cave";
import { theEnd, wedding } from "./finale";
import { chalk, dance, jars, merchant } from "./house";
import type { SceneBuilder } from "./kit";
import { riders, sesame, title, woodcutter } from "./opening";

export const SCENES: Record<SceneId, SceneBuilder> = {
  title,
  woodcutter,
  riders,
  sesame,
  treasure,
  kassim,
  chalk,
  merchant,
  jars,
  dance,
  wedding,
  theEnd,
};
