import type { Vec } from "../engine/flight";
import type { SceneId } from "../story/pages";
import { STALK_HEIGHT } from "./timelines";

export type Platform = "ground" | "cloud" | "leaf";

export interface Station {
  at: Vec;
  platform: Platform;
  night: boolean;
}

export const STALK_BASE: Vec = { x: 46, y: 0, z: -3.5 };
export const SKY_Y = STALK_HEIGHT;
export const CLOUD_Y = STALK_HEIGHT - 4;
export const STALK_GIRTH = 6;

export const STATIONS: Record<SceneId, Station> = {
  cover: { at: { x: 0, y: 0, z: 0 }, platform: "ground", night: true },
  cottage: { at: { x: 11, y: 0, z: 0 }, platform: "ground", night: false },
  beans: { at: { x: 21, y: 0, z: 0 }, platform: "ground", night: false },
  window: { at: { x: 31, y: 0, z: 0 }, platform: "ground", night: true },
  beanstalk: { at: { x: 41.5, y: 0, z: 0 }, platform: "ground", night: false },
  clouds: { at: { x: 41.5, y: SKY_Y, z: 0 }, platform: "cloud", night: false },
  castle: { at: { x: 54, y: SKY_Y, z: 0 }, platform: "cloud", night: false },
  treasure: { at: { x: 66, y: SKY_Y, z: 0 }, platform: "cloud", night: false },
  chase: { at: { x: 46, y: 16, z: 3 }, platform: "leaf", night: false },
  theEnd: { at: { x: 50.5, y: 0, z: 0 }, platform: "ground", night: false },
};
