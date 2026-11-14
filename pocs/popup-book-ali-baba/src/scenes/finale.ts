import { Vector3 } from "three";
import { camel } from "../art/animals";
import { musician } from "../art/people";
import { canopy, city, endBanner, ground, hangingLantern, palm, skyBanner } from "../art/scenery";
import { Act } from "../engine/act";
import { Sil } from "../engine/silhouette";
import { loop, smooth } from "../engine/timeline";
import { cast, flip, framesFor, lampCage, type SceneBuilder, ShadowSpray } from "./kit";

export const wedding: SceneBuilder = () => {
  const act = new Act().tint("#fecdd3", "#e8890b");
  act.place(new Sil(city(1.7), 1.7, { density: 0.55 }), 0.4, 1.3);
  act.place(new Sil(canopy(), 1.2), 0, 0.06, { y: 3.72, kind: "drop" });
  const lamps = [-3, -1.5, 0, 1.5, 3].map((x, i) => act.place(new Sil(hangingLantern(), 0.7, { anchorY: 1 }), x, 0.1, { y: 3.95 - (i % 2) * 0.15, kind: "drop" }));
  act.place(new Sil(palm(2.5, 1, 31), 2.5), -3.6, 0.3);
  act.place(new Sil(ground("floor"), 0.5), 0, 0.04, { y: -0.12 });
  const son = cast.son();
  act.place(son.body, -0.55, 0.14, { rod: true });
  const bride = cast.morgiana();
  act.place(flip(bride.body), 0.55, 0.14, { rod: true });
  const ali = cast.ali();
  act.place(ali.body, -2.45, 0.2, { rod: true });
  const drummer = flip(new Sil(musician().canvas, 1.15));
  act.place(drummer, 2.85, 0.15);
  const confetti = new ShadowSpray(140, null, 1, 0.5, 0.35).addTo(act);
  lampCage(act, 0.42, 0.25);
  act.onUpdate((t, dt) => {
    const sway = Math.sin(t * 1.6);
    son.arm.root.rotation.z = 1.25 + sway * 0.12;
    bride.arm.root.rotation.z = 1.25 - sway * 0.12;
    son.body.root.rotation.z = sway * 0.03;
    bride.body.root.rotation.z = -sway * 0.03;
    son.body.root.position.y = Math.abs(Math.sin(t * 1.6)) * 0.05;
    bride.body.root.position.y = Math.abs(Math.cos(t * 1.6)) * 0.05;
    ali.arm.root.rotation.z = 2.4 + Math.sin(t * 2) * 0.25;
    drummer.root.position.y = Math.abs(Math.sin(t * 6)) * 0.03;
    lamps.forEach((l, i) => (l.root.rotation.z = Math.sin(t * 1.1 + i * 0.8) * 0.15));
    if (Math.random() < dt * 40) confetti.emit(new Vector3((Math.random() - 0.5) * 8, 4.9, -0.2 - Math.random() * 0.5), new Vector3((Math.random() - 0.5) * 0.3, -0.2, 0), 7, 0.06 + Math.random() * 0.04, 12);
    confetti.update(dt);
  });
  act.lampMotion = (t, out) => out.set(Math.sin(t * 0.35) * 0.5, 0, 0.2);
  return act;
};

export const theEnd: SceneBuilder = () => {
  const act = new Act().tint("#fde68a", "#f0607a");
  act.place(new Sil(skyBanner(5), 1.25), 0, 0.05, { y: 3.55, kind: "drop" });
  act.place(new Sil(ground("dunes"), 0.55), 0, 0.04, { y: -0.1 });
  act.place(new Sil(palm(2.6, 1, 41), 2.6), -3.55, 0.25);
  act.place(flip(new Sil(palm(2.2, 1, 43), 2.2)), 3.6, 0.3);
  act.place(new Sil(ground("dunes"), 0.7, { density: 0.62 }), 0.3, 0.62, { y: 0.55 });
  const camels = [0, 1, 2].map((i) => {
    const c = flip(new Sil(framesFor(2, (f) => camel(f, i === 0)), 0.78, { density: 0.62 }));
    act.place(c, 0, 0.6 + i * 0.02, { y: 0.95 });
    return c;
  });
  const banner = act.place(new Sil(endBanner(), 1.0), 0, 0.1, { y: 2.2, kind: "rise", delay: 0.4 });
  const ali = cast.ali();
  act.place(ali.body, -1.55, 0.14, { rod: true });
  const morgiana = cast.morgiana();
  act.place(flip(morgiana.body), 1.55, 0.14, { rod: true });
  act.onUpdate((t) => {
    camels.forEach((c, i) => {
      c.root.position.x = 5 - loop(t * 0.35 + i * 1.35, 10.5);
      c.show(Math.floor(t * 2.4 + i) % 2);
      c.root.position.y = Math.abs(Math.sin(t * 2.4 * Math.PI * 0.5 + i)) * 0.02;
    });
    const bow = smooth(loop(t, 5), 2.5, 3.1) * (1 - smooth(loop(t, 5), 3.9, 4.6));
    ali.body.root.rotation.z = -bow * 0.35;
    morgiana.body.root.rotation.z = -bow * 0.35;
    ali.arm.root.rotation.z = bow * 0.9;
    morgiana.arm.root.rotation.z = bow * 0.9;
    banner.root.position.y = Math.sin(t * 0.8) * 0.05;
  });
  act.lampMotion = (t, out) => out.set(0, Math.sin(t * 0.3) * 0.1, -0.2);
  return act;
};
