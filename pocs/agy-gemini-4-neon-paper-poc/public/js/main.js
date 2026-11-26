import { EDITIONS, cloneEdition, getEditionById, applyHeadlineEdit } from './editions.js';
import { NewspaperScene } from './paper-scene.js';
import { PressAudio } from './audio.js';
import { BroadsheetUI } from './ui.js';

async function waitForFonts() {
  if (document.fonts && document.fonts.ready) {
    await Promise.race([
      document.fonts.ready,
      new Promise((resolve) => setTimeout(resolve, 1400))
    ]);
  }
}

async function boot() {
  await waitForFonts();

  const state = {
    edition: cloneEdition(EDITIONS[0]),
    spread: 0,
    folded: false,
    loupe: false,
    warmLamp: true,
    sound: true
  };

  const audio = new PressAudio();
  const stage = document.getElementById('stage');

  let ui = null;

  const scene = new NewspaperScene(stage, {
    onHover: (info) => ui && ui.updateHover(info),
    onLoupeMove: (sample) => ui && ui.updateLoupe(sample),
    onArticleSelect: (page, article) => {
      if (ui) ui.openReader(page, article);
    },
    onRequestSpread: (nextSpread) => {
      goToSpread(nextSpread);
    }
  });

  ui = new BroadsheetUI({
    onSelectEdition: (id) => {
      state.edition = cloneEdition(getEditionById(id));
      state.spread = 0;
      scene.setEdition(state.edition);
      scene.setSpread(0);
      scene.triggerPressStamp();
      audio.playPressThump();
      ui.closeReader();
      ui.syncEdition(state.edition);
      ui.syncSpread(state.edition, state.spread);
    },
    onSelectSpread: (idx) => goToSpread(idx),
    onStepSpread: (delta) => goToSpread(state.spread + delta),
    onToggleFold: () => {
      state.folded = !state.folded;
      scene.setFolded(state.folded);
      audio.playPageFlip();
      ui.syncToggles(state);
    },
    onToggleLoupe: () => {
      state.loupe = !state.loupe;
      scene.setLoupe(state.loupe);
      audio.playLampClick();
      ui.syncToggles(state);
    },
    onToggleLamp: () => {
      state.warmLamp = !state.warmLamp;
      scene.setWarmLamp(state.warmLamp);
      audio.playLampClick();
      ui.syncToggles(state);
    },
    onToggleSound: () => {
      state.sound = !state.sound;
      audio.setEnabled(state.sound);
      ui.syncToggles(state);
    },
    onResetCamera: () => {
      scene.resetCamera();
    },
    onCloseReader: () => {
      scene.resetCamera();
    },
    onApplyPressEdit: (patch) => {
      state.edition = applyHeadlineEdit(state.edition, patch);
      scene.setEdition(state.edition);
      if (state.spread !== 0) {
        state.spread = 0;
        scene.setSpread(0);
      }
      scene.triggerPressStamp();
      audio.playPressThump();
      ui.syncEdition(state.edition);
      ui.syncSpread(state.edition, state.spread);
    },
    onResetEdition: () => {
      state.edition = cloneEdition(getEditionById(state.edition.id));
      scene.setEdition(state.edition);
      scene.triggerPressStamp();
      audio.playPressThump();
      ui.syncEdition(state.edition);
      ui.syncSpread(state.edition, state.spread);
    }
  });

  function goToSpread(target) {
    const next = Math.max(0, Math.min(4, target));
    if (next === state.spread) return;
    state.spread = next;
    scene.setSpread(next);
    audio.playPageFlip();
    ui.closeReader();
    ui.syncSpread(state.edition, state.spread);
  }

  ui.renderEditionTabs(EDITIONS, state.edition.id);
  scene.setEdition(state.edition);
  ui.syncEdition(state.edition);
  ui.syncSpread(state.edition, state.spread);
  ui.syncToggles(state);

  window.addEventListener('keydown', (e) => {
    const tag = e.target && e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowRight') goToSpread(state.spread + 1);
    else if (e.key === 'ArrowLeft') goToSpread(state.spread - 1);
    else if (e.key.toLowerCase() === 'f') ui.handlers.onToggleFold();
    else if (e.key.toLowerCase() === 'l') ui.handlers.onToggleLoupe();
    else if (e.key.toLowerCase() === 't') ui.handlers.onToggleLamp();
    else if (e.key.toLowerCase() === 'p') ui.togglePressDrawer();
    else if (e.key.toLowerCase() === 'r') scene.resetCamera();
    else if (e.key === 'Escape') {
      ui.closeReader();
      scene.resetCamera();
    }
  });
}

boot();
