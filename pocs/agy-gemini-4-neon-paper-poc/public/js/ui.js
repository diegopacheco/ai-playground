const SPREAD_LABELS = ['1', '2–3', '4–5', '6–7', '8'];

export class BroadsheetUI {
  constructor(handlers) {
    this.handlers = handlers;
    this.clippedIds = new Set();
    this.currentArticle = null;

    this.elMasthead = document.getElementById('hud-masthead');
    this.elSpreadLabel = document.getElementById('hud-spread-label');
    this.elEditionTabs = document.getElementById('edition-tabs');
    this.elSpreadDots = document.getElementById('spread-dots');
    this.elHoverPill = document.getElementById('hover-pill');

    this.elLoupeHud = document.getElementById('loupe-hud');
    this.elLoupeCanvas = document.getElementById('loupe-canvas');
    this.elLoupeBadge = document.getElementById('loupe-badge');
    this.loupeCtx = this.elLoupeCanvas.getContext('2d');

    this.elReader = document.getElementById('reader-drawer');
    this.elReaderKicker = document.getElementById('reader-kicker');
    this.elReaderHeadline = document.getElementById('reader-headline');
    this.elReaderByline = document.getElementById('reader-byline');
    this.elReaderBody = document.getElementById('reader-body');
    this.elClipCount = document.getElementById('clip-count');
    this.btnClip = document.getElementById('btn-clip');

    this.elPress = document.getElementById('press-drawer');
    this.elPressForm = document.getElementById('press-form');
    this.inputMasthead = document.getElementById('input-masthead');
    this.inputBanner = document.getElementById('input-banner');
    this.inputDeck = document.getElementById('input-deck');
    this.inputLead = document.getElementById('input-lead');

    this.btnPrev = document.getElementById('btn-prev');
    this.btnNext = document.getElementById('btn-next');
    this.btnFold = document.getElementById('btn-fold');
    this.btnLoupe = document.getElementById('btn-loupe');
    this.btnReset = document.getElementById('btn-reset');
    this.btnLamp = document.getElementById('btn-lamp');
    this.btnSound = document.getElementById('btn-sound');
    this.btnPress = document.getElementById('btn-press');

    this.buildSpreadDots();
    this.bindControls();
  }

  renderEditionTabs(editions, activeId) {
    this.elEditionTabs.innerHTML = '';
    for (const ed of editions) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `edition-tab${ed.id === activeId ? ' active' : ''}`;
      btn.textContent = ed.label;
      btn.dataset.editionId = ed.id;
      btn.addEventListener('click', () => this.handlers.onSelectEdition(ed.id));
      this.elEditionTabs.appendChild(btn);
    }
  }

  buildSpreadDots() {
    this.elSpreadDots.innerHTML = '';
    SPREAD_LABELS.forEach((label, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'spread-dot';
      btn.textContent = label;
      btn.dataset.spread = String(idx);
      btn.addEventListener('click', () => this.handlers.onSelectSpread(idx));
      this.elSpreadDots.appendChild(btn);
    });
  }

  bindControls() {
    this.btnPrev.addEventListener('click', () => this.handlers.onStepSpread(-1));
    this.btnNext.addEventListener('click', () => this.handlers.onStepSpread(1));
    this.btnFold.addEventListener('click', () => this.handlers.onToggleFold());
    this.btnLoupe.addEventListener('click', () => this.handlers.onToggleLoupe());
    this.btnReset.addEventListener('click', () => this.handlers.onResetCamera());
    this.btnLamp.addEventListener('click', () => this.handlers.onToggleLamp());
    this.btnSound.addEventListener('click', () => this.handlers.onToggleSound());
    this.btnPress.addEventListener('click', () => this.togglePressDrawer());

    document.getElementById('reader-close').addEventListener('click', () => {
      this.closeReader();
      this.handlers.onCloseReader();
    });

    document.getElementById('press-close').addEventListener('click', () => {
      this.elPress.classList.add('hidden');
      this.btnPress.classList.remove('active');
    });

    this.btnClip.addEventListener('click', () => {
      if (!this.currentArticle) return;
      this.clippedIds.add(this.currentArticle.id);
      this.elClipCount.textContent = `${this.clippedIds.size} clipped`;
      this.btnClip.textContent = 'Clipped';
    });

    this.elPressForm.addEventListener('submit', (e) => {
      e.preventDefault();
      this.handlers.onApplyPressEdit({
        masthead: this.inputMasthead.value,
        banner: this.inputBanner.value,
        deck: this.inputDeck.value,
        leadBody: this.inputLead.value
      });
    });

    document.getElementById('btn-press-reset').addEventListener('click', () => {
      this.handlers.onResetEdition();
    });
  }

  syncEdition(edition) {
    this.elMasthead.textContent = edition.masthead;
    const front = edition.pages[0];
    const lead = front.articles.find((a) => a.lead) || front.articles[0];
    this.inputMasthead.value = edition.masthead;
    this.inputBanner.value = front.banner;
    this.inputDeck.value = front.deck;
    this.inputLead.value = lead ? lead.paragraphs.join('\n\n') : '';
    const tabs = this.elEditionTabs.querySelectorAll('.edition-tab');
    tabs.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.editionId === edition.id);
    });
  }

  syncSpread(edition, spread) {
    let text = '';
    if (spread === 0) {
      text = `PAGE 1 • ${edition.pages[0].section}`;
    } else if (spread === 4) {
      text = `PAGE 8 • ${edition.pages[7].section}`;
    } else {
      const left = edition.pages[spread * 2 - 1];
      const right = edition.pages[spread * 2];
      text = `PAGES ${left.number}–${right.number} • ${left.section}`;
    }
    this.elSpreadLabel.textContent = text;
    this.btnPrev.disabled = spread === 0;
    this.btnNext.disabled = spread === 4;
    this.elSpreadDots.querySelectorAll('.spread-dot').forEach((btn, idx) => {
      btn.classList.toggle('active', idx === spread);
    });
  }

  syncToggles({ folded, loupe, warmLamp, sound }) {
    this.btnFold.classList.toggle('active', folded);
    this.btnFold.innerHTML = `${folded ? 'Unfold' : 'Fold'} <kbd>F</kbd>`;
    this.btnLoupe.classList.toggle('active', loupe);
    this.elLoupeHud.classList.toggle('hidden', !loupe);
    this.btnLamp.innerHTML = `${warmLamp ? 'Banker Lamp' : 'Daylight'} <kbd>T</kbd>`;
    this.btnSound.textContent = sound ? 'Sound On' : 'Muted';
  }

  updateHover(info) {
    if (!info) {
      this.elHoverPill.classList.add('hidden');
      return;
    }
    if (info.corner === 'next') {
      this.elHoverPill.textContent = 'Click right corner to turn page →';
      this.elHoverPill.classList.remove('hidden');
      return;
    }
    if (info.corner === 'prev') {
      this.elHoverPill.textContent = '← Click left corner to turn back';
      this.elHoverPill.classList.remove('hidden');
      return;
    }
    if (info.article) {
      this.elHoverPill.textContent = `PAGE ${info.pageNumber} • ${info.article.headline} — Click to read`;
      this.elHoverPill.classList.remove('hidden');
      return;
    }
    this.elHoverPill.classList.add('hidden');
  }

  updateLoupe({ canvas, u, v, pageNumber }) {
    if (!canvas) return;
    const size = this.elLoupeCanvas.width;
    const sampleW = 340;
    const sampleH = 340;
    const sx = Math.max(0, Math.min(canvas.width - sampleW, u * canvas.width - sampleW / 2));
    const sy = Math.max(0, Math.min(canvas.height - sampleH, v * canvas.height - sampleH / 2));

    this.loupeCtx.clearRect(0, 0, size, size);
    this.loupeCtx.drawImage(canvas, sx, sy, sampleW, sampleH, 0, 0, size, size);
    this.elLoupeBadge.textContent = `PAGE ${pageNumber} • 2.8x OPTICAL LOUPE`;
  }

  openReader(page, article) {
    this.currentArticle = article;
    this.elReaderKicker.textContent = `PAGE ${page.number} • ${article.kicker}`;
    this.elReaderHeadline.textContent = article.headline;
    this.elReaderByline.textContent = `${article.byline} — ${page.section}`;
    this.elReaderBody.innerHTML = '';
    for (const p of article.paragraphs) {
      const node = document.createElement('p');
      node.textContent = p;
      this.elReaderBody.appendChild(node);
    }
    this.btnClip.textContent = this.clippedIds.has(article.id) ? 'Clipped' : 'Clip Article';
    this.elReader.classList.remove('hidden');
  }

  closeReader() {
    this.elReader.classList.add('hidden');
    this.currentArticle = null;
  }

  togglePressDrawer() {
    const willOpen = this.elPress.classList.contains('hidden');
    this.elPress.classList.toggle('hidden', !willOpen);
    this.btnPress.classList.toggle('active', willOpen);
  }
}
