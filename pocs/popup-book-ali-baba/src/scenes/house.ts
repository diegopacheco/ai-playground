import { donkey } from "../art/animals";
import { musician, thiefHead } from "../art/people";
import { city, colonnade, ground, hangingLantern, house, palm, scroll, skyBanner, veil } from "../art/scenery";
import { Act } from "../engine/act";
import { Sil } from "../engine/silhouette";
import { clamp01, loop, smooth } from "../engine/timeline";
import { cast, flip, framesFor, jar, type SceneBuilder, walkBob } from "./kit";
import { DAGGER_LOOP, danceOrbit, daggerShown, DOORS, doorMarks, JAR_X, lanternX, lidLift, morgianaX, thiefX, THIEF_MARK_AT, STREET_LOOP } from "./timelines";

export const chalk: SceneBuilder = () => {
  const act = new Act().tint("#a5b4fc", "#1b1650");
  act.place(new Sil(skyBanner(2), 1.25), 0, 0.05, { y: 3.55, kind: "drop" });
  act.place(new Sil(ground("street"), 0.5), 0, 0.04, { y: -0.12 });
  const houses = DOORS.map((x, i) => {
    const h = new Sil(framesFor(2, (f) => house(f === 1, i)), 2.2, { density: 0.68 });
    act.place(h, x, 0.3);
    return h;
  });
  const thief = cast.thief("none");
  act.place(thief.body, 0, 0.14, { rod: true });
  const morgiana = cast.morgiana();
  act.place(morgiana.body, 0, 0.16, { rod: true });
  act.onUpdate((t) => {
    const lt = loop(t, STREET_LOOP);
    thief.body.root.position.x = thiefX(t);
    walkBob(thief.body, t, 2.6, 0.06);
    thief.arm.root.rotation.z = 2.1 * smooth(lt, THIEF_MARK_AT - 0.6, THIEF_MARK_AT - 0.1) * (1 - smooth(lt, THIEF_MARK_AT + 0.3, THIEF_MARK_AT + 0.8));
    const mx = morgianaX(t);
    morgiana.body.root.position.x = mx;
    walkBob(morgiana.body, t, 3.2, 0.035);
    const near = Math.min(...DOORS.map((d) => Math.abs(mx + 0.45 - d)));
    morgiana.arm.root.rotation.z = 2.0 * clamp01(1 - near / 0.3);
    doorMarks(t).forEach((marked, i) => houses[i]?.show(marked ? 1 : 0));
  });
  act.lampMotion = (t, out) => out.set(Math.sin(t * 0.25) * 0.3, 0.2, -0.3);
  return act;
};

export const merchant: SceneBuilder = () => {
  const act = new Act().tint("#fdba74", "#5b21b6");
  act.place(new Sil(city(1.9), 1.9, { density: 0.55 }), -0.6, 1.2);
  act.place(new Sil(palm(2.4, 1, 21), 2.4), -3.5, 0.35);
  act.place(new Sil(ground("street"), 0.5), 0, 0.04, { y: -0.12 });
  act.place(flip(new Sil(house(false, 0), 2.2, { density: 0.72 })), 3.15, 0.3);
  const ali = cast.ali();
  act.place(flip(ali.body), 1.95, 0.16, { rod: true });
  const captain = cast.merchant();
  act.place(captain.body, 0, 0.14, { rod: true });
  const mules = [0, 1, 2].map((i) => {
    const m = new Sil(framesFor(2, (f) => donkey(f, "jars", 1.15)), 1.15);
    act.place(m, 0, 0.22 + i * 0.12, { rod: true });
    return m;
  });
  const words = new Sil(scroll("A bed for the night?", 300), 0.62);
  act.place(words, -0.2, 0.08, { y: 2.6, kind: "rise" });
  act.onUpdate((t) => {
    const walk = smooth(t, 0.5, 7.5);
    const moving = walk > 0 && walk < 1;
    const x = -6 + walk * 6.4;
    captain.body.root.position.x = x;
    if (moving) walkBob(captain.body, t, 3.4, 0.04);
    mules.forEach((m, i) => {
      m.root.position.x = x - 1.2 - i * 1.15;
      if (moving) m.show(Math.floor(t * 3 + i) % 2);
      m.root.position.y = moving ? Math.abs(Math.sin(t * 5 + i)) * 0.03 : 0;
    });
    const greet = smooth(t, 7.2, 8);
    captain.arm.root.rotation.z = greet * (1.3 + Math.sin(t * 2) * 0.15);
    ali.arm.root.rotation.z = greet * (1.8 + Math.sin(t * 5) * 0.5);
    words.root.scale.setScalar(Math.max(0.0001, smooth(t, 8, 8.6)));
  });
  act.lampMotion = (t, out) => out.set(Math.sin(t * 0.2) * 0.3, 0, 0);
  return act;
};

export const jars: SceneBuilder = () => {
  const act = new Act().tint("#7dd3fc", "#0b3a5c");
  act.place(new Sil(colonnade(), 3.4, { density: 0.7 }), 0, 0.75);
  act.place(new Sil(ground("floor"), 0.5), 0, 0.04, { y: -0.12 });
  const pots = JAR_X.map((x, i) => {
    const head = new Sil(thiefHead().canvas, 0.42);
    act.place(head, x + 0.02, 0.5, { kind: "rise", y: 0.7 });
    const j = jar(1.05);
    act.solid(j.group, x, 0.46, 0, 0.1 + i * 0.05);
    return { head, lid: j.lid };
  });
  const whisper = new Sil(scroll("Is it time?", 160), 0.6);
  act.place(whisper, 0, 0.06, { y: 2.65, kind: "rise" });
  const morgiana = cast.morgiana();
  act.place(morgiana.body, 0, 0.14, { rod: true });
  const light = new Sil(hangingLantern(), 0.62, { anchorY: 1 });
  morgiana.arm.attach(light, morgiana.hand[0], morgiana.hand[1]);
  act.onUpdate((t, dt) => {
    let loudest = -1;
    let level = 0.55;
    pots.forEach((pot, i) => {
      const lift = lidLift(t, i);
      pot.lid.position.y = 1.17 + lift * 0.2;
      pot.lid.rotation.z = lift * 0.6;
      pot.head.root.position.y = lift * 0.75;
      if (lift > level) {
        level = lift;
        loudest = i;
      }
    });
    const target = loudest >= 0 ? 1 : 0;
    if (loudest >= 0) whisper.root.position.x = JAR_X[loudest] ?? 0;
    const s = whisper.root.scale.x + (target - whisper.root.scale.x) * Math.min(dt * 8, 1);
    whisper.root.scale.setScalar(Math.max(s, 0.0001));
    const x = lanternX(t);
    morgiana.body.root.position.x = x - 0.55;
    walkBob(morgiana.body, t, 2.4, 0.03);
    morgiana.arm.root.rotation.z = 1.25 + Math.sin(t * 2.4) * 0.05;
    light.root.rotation.z = -morgiana.arm.root.rotation.z + Math.sin(t * 2.4) * 0.12;
  });
  act.lampMotion = (t, out) => out.set(lanternX(t) * 0.85, -0.35, 0.55);
  return act;
};

export const dance: SceneBuilder = () => {
  const act = new Act().tint("#f0abfc", "#5b1a8f");
  act.place(new Sil(colonnade(), 3.4, { density: 0.7 }), 0, 0.8);
  act.place(new Sil(ground("floor"), 0.5), 0, 0.04, { y: -0.12 });
  const lamps = [-2.2, 0, 2.2].map((x) => act.place(new Sil(hangingLantern(), 0.8, { anchorY: 1 }), x, 0.2, { y: 4.6, kind: "drop" }));
  act.place(new Sil(musician().canvas, 1.15), -3.15, 0.15);
  const captain = cast.merchant("dagger");
  act.place(flip(captain.body), 2.45, 0.16, { rod: true });
  const ali = cast.ali();
  act.place(flip(ali.body), 3.45, 0.22, { rod: true });
  const morgiana = cast.morgiana();
  act.place(morgiana.body, -0.2, 0.12, { rod: true });
  const scarf = new Sil(veil(), 1.1, { anchorY: 1, density: 0.55 });
  morgiana.arm.attach(scarf, morgiana.hand[0], morgiana.hand[1]);
  const base = act.top.clone();
  act.onUpdate((t) => {
    const lt = loop(t, DAGGER_LOOP);
    const spin = Math.PI * 2 * (smooth(lt, 2.2, 3.6) + smooth(lt, 6.4, 7.6));
    morgiana.body.root.rotation.y = spin;
    morgiana.body.root.position.x = Math.sin(t * 0.9) * 0.9;
    morgiana.body.root.position.y = Math.abs(Math.sin(t * 3.2)) * 0.08;
    morgiana.arm.root.rotation.z = 2.3 + Math.sin(t * 3.2) * 0.5;
    scarf.root.rotation.z = Math.sin(t * 3.2 + 1) * 0.6 - 0.3;
    lamps.forEach((l, i) => (l.root.rotation.z = Math.sin(t * 1.3 + i) * 0.12));
    const shown = daggerShown(t);
    captain.arm.root.rotation.z += ((shown ? 2.3 : 0.05) - captain.arm.root.rotation.z) * 0.1;
    ali.arm.root.rotation.z = shown ? 1.4 : 0.1 + Math.sin(t * 3.2) * 0.1;
    const alarm = smooth(lt, 7, 7.4) * (1 - smooth(lt, 9.2, 9.8));
    act.top.copy(base).offsetHSL(Math.sin(t * 0.4) * 0.05, 0, 0).lerp(act.bottom.clone().set("#ff3b3b"), alarm * 0.75);
  });
  act.lampMotion = (t, out) => {
    const o = danceOrbit(t);
    out.set(o.x, Math.sin(t * 1.1) * 0.15, o.z);
  };
  return act;
};
