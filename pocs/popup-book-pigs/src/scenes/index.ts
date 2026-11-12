import type { SceneId } from "../story/pages";
import { pot, roof, theEnd } from "./finale";
import type { SceneBuilder } from "./kit";
import { bricks, cover, farewell, sticks, straw } from "./opening";
import { blowSticks, blowStraw, brickStands, knock } from "./wolf";

export const SCENES: Record<SceneId, SceneBuilder> = {
  cover,
  farewell,
  straw,
  sticks,
  bricks,
  knock,
  blowStraw,
  blowSticks,
  brickStands,
  roof,
  pot,
  theEnd,
};
