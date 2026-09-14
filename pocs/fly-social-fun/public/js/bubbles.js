import * as THREE from 'three';
import { escapeHtml, renderBuzzHtml } from './format.js';

const BUZZ_MS = 5200;
const BURST_MS = 1100;
const MAX_BUBBLES = 6;
const LIFT = 0.9;
const GAP = 6;

export function createBubbles(layer, camera, positionOf) {
  const active = [];
  const projected = new THREE.Vector3();

  function track(item) {
    layer.append(item.el);
    active.push(item);
  }

  function drop(item) {
    active.splice(active.indexOf(item), 1);
    item.el.remove();
  }

  function show(buzz) {
    const existing = active.find((item) => item.flyId === buzz.flyId);
    if (existing) drop(existing);
    const bubbles = active.filter((item) => item.flyId);
    if (bubbles.length >= MAX_BUBBLES) drop(bubbles[0]);
    const el = document.createElement('div');
    el.className = 'bubble';
    el.innerHTML = `<strong>${escapeHtml(buzz.name)}</strong><span>${renderBuzzHtml(buzz.text)}</span>`;
    track({ el, flyId: buzz.flyId, until: performance.now() + BUZZ_MS });
  }

  function burst(position, word) {
    const el = document.createElement('div');
    el.className = 'burst';
    el.textContent = word;
    track({ el, position: position.clone(), until: performance.now() + BURST_MS });
  }

  function update() {
    const now = performance.now();
    const width = layer.clientWidth;
    const height = layer.clientHeight;
    const placed = [];
    for (const item of [...active]) {
      const position = item.position || positionOf(item.flyId);
      if (!position || now > item.until) {
        drop(item);
        continue;
      }
      projected.copy(position);
      projected.y += LIFT;
      projected.project(camera);
      const visible = projected.z < 1;
      item.el.style.visibility = visible ? 'visible' : 'hidden';
      if (!visible) continue;
      item.x = (projected.x * 0.5 + 0.5) * width;
      item.y = (-projected.y * 0.5 + 0.5) * height;
      placed.push(item);
    }
    spread(placed.filter((item) => item.flyId));
    for (const item of placed) item.el.style.translate = `${item.x}px ${item.y}px`;
  }

  function spread(items) {
    const settled = [];
    items.sort((a, b) => b.y - a.y);
    for (const item of items) {
      item.w ||= item.el.offsetWidth;
      item.h ||= item.el.offsetHeight;
      let moved = true;
      while (moved) {
        moved = false;
        for (const other of settled) {
          const overlapX = Math.abs(item.x - other.x) < (item.w + other.w) / 2 + GAP;
          const overlapY = item.y > other.y - other.h - GAP && item.y - item.h < other.y + GAP;
          if (overlapX && overlapY) {
            item.y = other.y - other.h - GAP;
            moved = true;
          }
        }
      }
      settled.push(item);
    }
  }

  return { show, burst, update };
}
