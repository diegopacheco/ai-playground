import { Vector3 } from "three";
import { chest, cliff, ground, oilLamp, sack, scroll, stalactites, treasureHeap } from "../art/scenery";
import { Act } from "../engine/act";
import { Sil } from "../engine/silhouette";
import { loop, smooth } from "../engine/timeline";
import { boulder, cast, coins, flip, lampCage, type SceneBuilder } from "./kit";
import { WRONG_WORDS, wrongWord } from "./timelines";

function caveRoof(act: Act): void {
  act.place(new Sil(stalactites(), 1.4), 0, 0.06, { y: 3.5, kind: "drop" });
  act.place(new Sil(ground("hills"), 0.5), 0, 0.04, { y: -0.12 });
}

export const treasure: SceneBuilder = () => {
  const act = new Act().tint("#ffd166", "#c2610a");
  caveRoof(act);
  act.place(new Sil(treasureHeap(4), 1.5), -2.55, 0.3);
  act.place(flip(new Sil(treasureHeap(9), 1.7)), 2.75, 0.45);
  act.place(new Sil(oilLamp(), 0.42), -2.6, 0.26, { y: 1.25, kind: "rise" });
  act.place(new Sil(sack(), 0.65), -1.25, 0.12);
  act.place(new Sil(sack(), 0.55), 1.95, 0.1);
  const box = act.place(new Sil(chest(true), 0.8), 0.95, 0.1);
  const ali = cast.ali();
  act.place(ali.body, -0.35, 0.15, { rod: true });
  const spray = coins(70).addTo(act);
  lampCage(act, 0.5, 0.35);
  act.onUpdate((t, dt) => {
    ali.arm.root.rotation.z = 1.6 + Math.sin(t * 1.4) * 0.6;
    ali.body.root.position.y = Math.abs(Math.sin(t * 2.8)) * 0.04;
    ali.body.root.rotation.z = Math.sin(t * 0.7) * 0.03;
    box.root.rotation.z = Math.sin(t * 9) * 0.006;
    if (Math.random() < dt * 22) {
      spray.emit(new Vector3(0.95 + (Math.random() - 0.5) * 0.4, 0.75, -0.1), new Vector3((Math.random() - 0.5) * 1.6, 2.4 + Math.random() * 1.3, (Math.random() - 0.5) * 0.3), 1.6, 0.13 + Math.random() * 0.05, 9);
    }
    spray.update(dt);
  });
  act.lampMotion = (t, out) => out.set(Math.sin(t * 0.4) * 0.4, Math.sin(t * 0.6) * 0.1, 0.3);
  return act;
};

export const kassim: SceneBuilder = () => {
  const act = new Act().tint("#fb7185", "#6b1020");
  caveRoof(act);
  act.place(new Sil(cliff(3.5), 3.5), 2.75, 0.35);
  act.solid(boulder(0.6), 2.77, 0.55, 0.62);
  act.place(new Sil(treasureHeap(13), 1.3, { density: 0.7 }), -3.1, 0.5);
  [-2.6, -2.0, -1.45].forEach((x, i) => act.place(new Sil(sack(), 0.6 - i * 0.05), x, 0.1 + i * 0.05));
  const kassimP = cast.kassim();
  act.place(kassimP.body, -0.55, 0.14, { rod: true });
  const words = new Sil(WRONG_WORDS.map((w) => scroll(w, 220)), 0.62);
  act.place(words, -0.45, 0.08, { y: 2.6, kind: "rise" });
  const captain = cast.captain("scimitar");
  act.place(captain.body, -3.6, 0.2);
  let shown = -1;
  let pop = 1;
  act.onUpdate((t, dt) => {
    kassimP.arm.root.rotation.z = 1.9 + Math.sin(t * 7) * 0.7;
    kassimP.body.root.position.y = Math.abs(Math.sin(t * 3.6)) * 0.08;
    kassimP.body.root.rotation.z = Math.sin(t * 3.6) * 0.04;
    const word = wrongWord(t);
    if (word !== shown) {
      shown = word;
      words.show(word);
      pop = 0;
    }
    pop = Math.min(1, pop + dt * 3);
    words.root.scale.setScalar(0.6 + 0.4 * (1 - Math.pow(1 - pop, 3)) + Math.sin(pop * Math.PI) * 0.12);
    const lt = loop(t, 12);
    const loom = smooth(lt, 6, 9) * (1 - smooth(lt, 10.4, 11.9));
    captain.body.root.position.set(loom * 2.2, 0, -loom * 2.65);
    captain.arm.root.rotation.z = 0.2 + loom * 1.9;
  });
  act.levelAt = (t) => {
    const beat = loop(t, 1.1);
    return 1 - 0.14 * (Math.exp(-beat * 18) + Math.exp(-Math.max(beat - 0.22, 0) * 18) * (beat > 0.22 ? 1 : 0));
  };
  act.lampMotion = (t, out) => out.set(0, 0, -0.2 + Math.sin(t * 0.5) * 0.2);
  return act;
};
