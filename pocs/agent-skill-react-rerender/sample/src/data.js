function seeded(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const rand = seeded(20260614);
const pick = (arr) => arr[Math.floor(rand() * arr.length)];

const names = [
  "Aurora Headset", "Nimbus Router", "Cobalt Keyboard", "Lumen Lamp", "Pulse Watch",
  "Drift Speaker", "Ember Mug", "Quartz Mouse", "Halo Webcam", "Vertex Stand",
  "Glide Trackpad", "Nova Charger", "Slate Tablet", "Comet Drive", "Prism Monitor",
  "Atlas Backpack", "Vapor Bottle", "Echo Earbuds", "Cinder Grill", "Tidal Speaker",
  "Onyx Stylus", "Solstice Dock", "Mistral Fan", "Beacon Light",
];

const categories = ["Audio", "Peripherals", "Wearables", "Home", "Storage", "Display"];
const regions = ["NA", "EU", "APAC", "LATAM"];
const actors = ["sofia", "marcus", "lena", "kenji", "priya", "diego", "noah", "amara"];
const verbs = ["purchased", "refunded", "viewed", "wishlisted", "reviewed", "returned"];

export const products = names.map((name, i) => ({
  id: i + 1,
  name,
  category: pick(categories),
  price: Math.round(1900 + rand() * 38000) / 100,
  stock: Math.floor(rand() * 480),
  rating: Math.round((3 + rand() * 2) * 10) / 10,
  trend: Math.round((rand() * 40 - 12) * 10) / 10,
}));

export const activities = Array.from({ length: 240 }, (_, i) => ({
  id: i + 1,
  ts: Date.now() - Math.floor(rand() * 1000 * 60 * 60 * 72),
  actor: pick(actors),
  verb: pick(verbs),
  product: pick(names),
  amount: Math.round(rand() * 42000) / 100,
  region: pick(regions),
}));

export const metrics = Array.from({ length: 500 }, (_, i) => ({
  id: i + 1,
  sku: "SKU-" + String(1000 + i),
  region: pick(regions),
  category: pick(categories),
  units: Math.floor(rand() * 5200),
  revenue: Math.round(rand() * 980000) / 100,
  margin: Math.round((rand() * 0.6 + 0.05) * 1000) / 1000,
  refunds: Math.floor(rand() * 140),
}));

export const kpis = {
  revenue: metrics.reduce((a, m) => a + m.revenue, 0),
  units: metrics.reduce((a, m) => a + m.units, 0),
  orders: activities.filter((a) => a.verb === "purchased").length,
  refundRate: activities.filter((a) => a.verb === "refunded").length / activities.length,
};
