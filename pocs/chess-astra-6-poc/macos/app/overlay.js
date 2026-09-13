(() => {
  if (document.getElementById('mac-strip')) return;
  const game = location.protocol.startsWith('http');
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const svg = (d, color) => `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="${color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${d}"/></svg>`;

  function strip() {
    const bar = element('div', '', '');
    bar.id = 'mac-strip';
    bar.innerHTML = '<span>The Wizard’s Gambit</span>';
    bar.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      bar.setPointerCapture(event.pointerId);
      window.macApp.dragStart(event.screenX, event.screenY);
    });
    bar.addEventListener('pointermove', event => {
      if (bar.hasPointerCapture(event.pointerId)) window.macApp.dragMove(event.screenX, event.screenY);
    });
    bar.addEventListener('dblclick', () => window.macApp.toggleMaximize());
    document.body.append(bar);
    document.documentElement.classList.add('mac-app');
  }

  function modal(id, label) {
    const backdrop = element('div', 'mac-backdrop');
    backdrop.id = id;
    backdrop.hidden = true;
    const card = element('section', 'mac-card');
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-modal', 'true');
    card.setAttribute('aria-label', label);
    const input = element('input', 'mac-search');
    input.type = 'search';
    input.spellcheck = false;
    input.autocomplete = 'off';
    input.setAttribute('aria-label', label);
    card.append(input);
    backdrop.append(card);
    document.body.append(backdrop);
    const close = () => { backdrop.hidden = true; };
    backdrop.addEventListener('mousedown', event => { if (event.target === backdrop) close(); });
    input.addEventListener('keydown', event => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      event.stopPropagation();
      if (input.value) {
        input.value = '';
        input.dispatchEvent(new Event('input'));
      } else close();
    });
    const open = () => {
      for (const other of document.querySelectorAll('.mac-backdrop')) other.hidden = true;
      backdrop.hidden = false;
      input.value = '';
      input.dispatchEvent(new Event('input'));
      input.focus();
    };
    return { backdrop, card, input, open, close };
  }

  function choose(select, value) {
    select.value = value;
    select.dispatchEvent(new Event('change'));
  }

  function searchItems() {
    const items = [];
    const click = (selector, group) => {
      const button = document.querySelector(selector);
      if (!button || button.disabled) return;
      const label = button.getAttribute('aria-label') || button.textContent.replace(/[↗→]/g, '').trim();
      items.push({ group, label, run: () => button.click() });
    };
    click('#new-game', 'Game');
    click('#undo', 'Game');
    click('#guide-button', 'Game');
    const input = document.getElementById('move-input');
    if (input && !input.disabled) items.push({ group: 'Game', label: 'Enter a move', run: () => input.focus() });
    for (const selector of ['#flip', '#view', '#sound', '#fullscreen']) click(selector, 'Board');
    for (const [id, group] of [['difficulty', 'Challenge'], ['piece-style', 'Piece style'], ['background', 'Room']]) {
      const select = document.getElementById(id);
      for (const option of select?.options ?? []) items.push({ group, label: option.textContent, current: option.selected, run: () => choose(select, option.value) });
    }
    for (const radio of document.querySelectorAll('[name="piece-color"]')) {
      items.push({ group: 'Piece color', label: radio.value[0].toUpperCase() + radio.value.slice(1), current: radio.checked, run: () => { radio.checked = true; radio.dispatchEvent(new Event('change')); } });
    }
    items.push({ group: 'Help', label: 'Keyboard shortcuts', run: () => shortcuts.open() });
    return items;
  }

  function palette() {
    const view = modal('mac-palette', 'Search anything');
    view.input.placeholder = 'Search moves, rooms, colors, styles, board controls…';
    const list = element('ul', 'mac-results');
    list.setAttribute('role', 'listbox');
    const footer = element('p', 'mac-footer', '↑ ↓ to move · ↩ to go · Esc to clear or close');
    view.card.append(list, footer);
    let items = [];
    let shown = [];
    let active = 0;
    const draw = () => {
      list.replaceChildren(...shown.map((item, index) => {
        const row = element('li', index === active ? 'active' : '');
        row.setAttribute('role', 'option');
        row.setAttribute('aria-selected', String(index === active));
        row.append(element('span', 'mac-result-label', item.label), element('span', 'mac-result-group', item.current ? `${item.group} · current` : item.group));
        row.addEventListener('mousemove', () => { if (active !== index) { active = index; draw(); } });
        row.addEventListener('click', () => go(item));
        return row;
      }));
      if (!shown.length) list.replaceChildren(element('li', 'mac-empty', `Nothing matches “${view.input.value}”.`));
      list.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
    };
    const go = item => { view.close(); item.run(); };
    view.input.addEventListener('input', () => {
      const words = view.input.value.toLowerCase().split(/\s+/).filter(Boolean);
      shown = items.filter(item => words.every(word => `${item.group} ${item.label}`.toLowerCase().includes(word)));
      active = 0;
      draw();
    });
    view.input.addEventListener('keydown', event => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        active = (active + (event.key === 'ArrowDown' ? 1 : -1) + shown.length) % Math.max(shown.length, 1);
        draw();
      } else if (event.key === 'Enter' && shown[active]) {
        event.preventDefault();
        go(shown[active]);
      }
    });
    return { ...view, open: () => { items = searchItems(); view.open(); } };
  }

  function shortcutSheet() {
    const view = modal('mac-shortcuts', 'Keyboard shortcuts');
    view.card.classList.add('mac-sheet');
    view.input.placeholder = 'Search shortcuts…';
    const count = element('p', 'mac-count');
    const columns = element('div', 'mac-groups');
    view.card.append(count, columns);
    const total = window.SHORTCUTS.reduce((sum, group) => sum + group.items.length, 0);
    view.input.addEventListener('input', () => {
      const query = view.input.value.trim().toLowerCase();
      let matched = 0;
      const groups = window.SHORTCUTS.map(group => {
        const whole = group.title.toLowerCase().includes(query);
        const rows = whole ? group.items : group.items.filter(item => `${item.keys.join(' ')} ${item.label}`.toLowerCase().includes(query));
        if (!rows.length) return null;
        matched += rows.length;
        const box = element('section', 'mac-group');
        box.style.setProperty('--group', group.color);
        const title = element('h3');
        title.innerHTML = svg(group.icon, group.color);
        title.append(element('span', '', group.title));
        box.append(title);
        for (const item of rows) {
          const row = element('div', 'mac-shortcut');
          const keys = element('span', 'mac-keys');
          keys.append(...item.keys.map(key => element('kbd', '', key)));
          row.append(element('span', '', item.label), keys);
          box.append(row);
        }
        return box;
      }).filter(Boolean);
      columns.replaceChildren(...groups);
      if (!groups.length) columns.append(element('p', 'mac-empty', `No shortcuts match “${view.input.value.trim()}”.`));
      count.textContent = query ? `${matched} of ${total} shortcuts` : `${total} shortcuts`;
    });
    return view;
  }

  function toast(text) {
    const note = document.getElementById('mac-toast') ?? document.body.appendChild(Object.assign(element('div'), { id: 'mac-toast' }));
    note.textContent = text;
    note.classList.add('visible');
    clearTimeout(note.timer);
    note.timer = setTimeout(() => note.classList.remove('visible'), 2600);
  }

  strip();
  const shortcuts = game ? shortcutSheet() : null;
  const search = game ? palette() : null;
  window.macApp.onCommand(({ type, value }) => {
    if (type === 'toast') toast(value);
    if (!game) return;
    if (type === 'search') search.open();
    if (type === 'shortcuts') shortcuts.open();
    if (type === 'room') {
      const select = document.getElementById('background');
      if (select) choose(select, value);
    }
  });
})();
