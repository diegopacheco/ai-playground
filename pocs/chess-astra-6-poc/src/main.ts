import './style.css';
import { GameAudio } from './audio';
import { Chess, type Square, type Move } from 'chess.js';
import { ChessScene, pieceColors, type PieceColor } from './scene';
import { restoreGame, skipStuckTurn, type Difficulty, type SearchReply } from './engine';
import { backgrounds, type Background } from './rooms';
import { pieceStyles, type PieceStyle } from './pieceStyles';

const icons = {
  spark: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z"/>',
  undo: '<path d="M9 5 4 10l5 5M4 10h10a6 6 0 0 1 0 12" transform="translate(0 -2)"/>',
  flip: '<path d="m7 3-4 4 4 4M3 7h13M17 21l4-4-4-4m4 4H8"/>',
  fullscreen: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',
  view: '<path d="m12 3 9 5v8l-9 5-9-5V8zM3 8l9 5 9-5m-9 5v8"/>',
  book: '<path d="M12 6v15M3 4c4-1 6 0 9 2 3-2 5-3 9-2v15c-4-1-6 0-9 2-3-2-5-3-9-2z"/>',
  sound: '<path d="M11 4 5 9H2v6h3l6 5zM15 8c3 2 3 6 0 8m3-11c5 4 5 10 0 14"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
};
const icon = (name: keyof typeof icons) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]}</svg>`;
const rooms: Record<Background, { label: string; caption: string }> = {
  library: { label: 'HOGWARTS · THE LIBRARY', caption: 'Among ancient books, a new story unfolds.' },
  greatHall: { label: 'HOGWARTS · THE GREAT HALL', caption: 'Beneath a thousand candles, the houses watch.' },
  office: { label: 'HOGWARTS · DUMBLEDORE’S OFFICE', caption: 'The portraits pretend to sleep. They are watching.' },
};
const options = (entries: Record<string, string>) => Object.entries(entries).map(([value, label]) => `<option value="${value}">${label}</option>`).join('');
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="masthead">
    <a class="brand" href="./" aria-label="The Wizard’s Gambit home"><img src="/logo.svg" alt="" /><span>THE WIZARD’S<br><strong>GAMBIT</strong></span></a>
    <div class="header-note"><span class="tiny-star">✧</span> A LITTLE MAGIC. A GAME OF MINDS.</div>
    <button class="quiet-button" id="guide-button">${icon('book')}<span>How to play</span></button>
  </header>
  <main>
    <div class="intro"><div><div class="eyebrow"><span></span> THE ENCHANTED CHESS CLUB</div><h1>Make your move.<br><em>Leave a little magic.</em></h1><p>A timeless game, in a world a little less ordinary.</p></div><div class="edition"><span>EST. IN ANOTHER REALM</span><div>32 pieces. Infinite possibilities.</div><span class="edition-line"></span></div></div>
    <div class="game-layout" id="game-layout">
      <section class="arena" aria-label="Chess table">
        <div class="arena-top"><span class="room-label"><span class="live-dot"></span> <span id="room-name">HOGWARTS · THE LIBRARY</span></span><span class="room-meta" id="music-status">HEDWIG’S THEME · SOUND OFF</span></div>
        <div class="music-panel" id="music-panel" hidden><p>Press play on the sequencer. This panel hides once you do.</p><iframe id="music-player" title="Hedwig’s Theme on Online Sequencer" allow="autoplay"></iframe></div>
        <div id="scene"><div class="scene-caption"><span>✧</span> <i id="room-caption">Among ancient books, a new story unfolds.</i></div></div>
        <section class="match-result" id="match-result" role="status" aria-labelledby="result-title" hidden>
          <span class="result-crown" aria-hidden="true">♔</span>
          <div class="eyebrow">THE DUEL IS DECIDED</div>
          <h2 id="result-title">Checkmate</h2>
          <p id="result-winner"></p>
          <button class="primary-button" id="result-new-game">${icon('spark')} New game ${icon('arrow')}</button>
        </section>
        <div class="board-toolbar"><div class="turn-pill" id="board-turn"><span class="live-dot"></span> Your move, wizard.</div><div class="camera-actions"><button id="flip" title="Rotate board" aria-label="Rotate board">${icon('flip')}</button><button id="view" title="Switch to overhead view" aria-label="Switch to overhead view">${icon('view')}</button><button id="sound" title="Play library music and sound" aria-label="Play library music and sound" aria-pressed="false">${icon('sound')}<span class="sound-off"></span></button><button id="fullscreen" title="Enter game fullscreen" aria-label="Enter game fullscreen" aria-pressed="false">${icon('fullscreen')}</button></div></div>
        <div class="board-bottom"><span>Drag to orbit <b>·</b> <i class="hint-mouse">Scroll to explore</i><i class="hint-touch">Pinch to zoom</i></span><form id="move-form"><label for="move-input">Move</label><input id="move-input" aria-label="Move notation" placeholder="e2e4" autocomplete="off" spellcheck="false" maxlength="8"/><button type="submit" aria-label="Play move">${icon('arrow')}</button></form></div>
      </section>
      <aside class="sidebar">
        <section class="match-card"><div class="section-label">YOUR OPPONENT <span>01</span></div><div class="opponent"><div class="avatar">♞<span>✧</span></div><div><h2>The Castle Guardian</h2><p>A worthy mind. An ancient magic.</p></div></div><label class="field-label" for="difficulty">CHOOSE YOUR CHALLENGE</label><div class="select-wrap"><select id="difficulty"><option value="apprentice">Apprentice</option><option value="wizard" selected>Wizard</option><option value="grandmaster">Grandmaster</option></select><span>⌄</span></div><p class="difficulty-note" id="difficulty-note">A thoughtful duel. Two moves deep.</p><fieldset class="piece-colors"><legend>YOUR PIECE COLOR</legend><div class="color-options">${Object.entries(pieceColors).map(([color, hex]) => `<label class="color-option"><input type="radio" name="piece-color" value="${color}" ${color === 'white' ? 'checked' : ''}/><span class="color-swatch" style="--swatch:${hex}"></span><span>${color[0].toUpperCase() + color.slice(1)}</span></label>`).join('')}</div><p>You move first · CPU: <span id="cpu-color">Green</span></p></fieldset><div class="look-options"><div><label class="field-label" for="piece-style">PIECE STYLE</label><div class="select-wrap"><select id="piece-style">${options(pieceStyles)}</select><span>⌄</span></div></div><div><label class="field-label" for="background">BACKGROUND</label><div class="select-wrap"><select id="background">${options(backgrounds)}</select><span>⌄</span></div></div></div><button class="primary-button" id="new-game">${icon('spark')} New game <span>↗</span></button></section>
        <section class="chronicle"><div class="section-label">THE CHRONICLE <span id="move-count">0 MOVES</span></div><div class="history-head"><span>TURN</span><span id="human-heading">WHITE</span><span id="cpu-heading">GREEN</span></div><div id="history" aria-label="Move history"><div class="empty-history"><span>♙</span><p>Every great story<br>begins with a bold move.</p><small>Your first chapter awaits.</small></div></div><button class="undo-button" id="undo" disabled>${icon('undo')} Take back a turn</button></section>
        <div class="status-card" role="status" aria-live="polite"><span class="status-spark">✧</span><div><strong id="status-title">The board is yours.</strong><p id="status-text">Select one of your pieces to see its possibilities.</p></div></div>
      </aside>
    </div>
    <footer><span>CRAFTED FOR THE LOVE OF THE GAME.</span><span class="footer-center">Strategy is the real magic.</span><span>HUMAN <span class="footer-cross">×</span> CPU</span></footer>
  </main>
  <dialog id="guide"><button class="dialog-close" aria-label="Close guide">×</button><div class="eyebrow">A SHORT SPELLBOOK</div><h2>The rules of the realm.</h2><p>Choose a color and a material for your pieces, and the room you play in: the Library, the Great Hall, or Dumbledore’s Office. The Guardian uses a contrasting color. You always move first; these choices only change appearance. Tap the speaker to play Hedwig’s Theme along with move, capture, and checkmate sounds, and a crackling fire in the Library.</p><ol><li>Select one of your pieces, then a golden ring to move. Drag the board to look around.</li><li>Protect your king and put the opposing king in checkmate. Legal moves, castling, and en passant are handled for you.</li><li>When a pawn reaches the far rank, choose its new piece. A check must be answered immediately.</li><li>There is no stalemate. A side that is not in check but has no legal move skips its turn, and the other side moves again.</li><li>Use the Move field with coordinates such as <strong>e2e4</strong>, or chess notation such as <strong>Nf3</strong>. Add q, r, b, or n for promotion.</li><li>Take back a turn to try another idea. Your match saves automatically in this browser when storage is available.</li></ol><p class="dialog-note">Apprentice searches 1 ply, Wizard 2, and Grandmaster up to 3 within a 1.4-second budget. This is a casual opponent, with no rating claim.</p><button class="primary-button dialog-done">Let the game begin ${icon('arrow')}</button></dialog>
  <dialog id="restart"><div class="eyebrow">A FRESH CHAPTER</div><h2>Begin a new game?</h2><p>Your current match will be replaced.</p><div class="dialog-actions"><button class="quiet-button" id="cancel-restart">Keep playing</button><button class="primary-button" id="confirm-restart">New game</button></div></dialog>
  <dialog id="promotion"><div class="eyebrow">A LITTLE TRANSFORMATION</div><h2>Choose your new piece.</h2><div class="promotion-options"><button data-piece="q">♕<span>Queen</span></button><button data-piece="r">♖<span>Rook</span></button><button data-piece="b">♗<span>Bishop</span></button><button data-piece="n">♘<span>Knight</span></button></div></dialog>
`;
const music = 'https://onlinesequencer.net/1073884';
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
let game = new Chess();
let selected: Square | null = null;
let difficulty: Difficulty = 'wizard';
let pieceColor: PieceColor = 'white';
let pieceStyle: PieceStyle = 'classic';
let background: Background = 'library';
let worker: Worker | null = null;
let thinking = false;
let cpuTimer: number | undefined;
let sound = false;
const audio = new GameAudio();
let scene: ChessScene | null = null;
let promotion: { from: Square; to: Square } | null = null;
const storageKey = 'wizards-gambit-v1';
try {
  const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
  if (saved && Array.isArray(saved.history) && saved.history.every((move: unknown) => typeof move === 'string')) {
    game = restoreGame(saved.history);
    skipStuckTurn(game);
    if (['apprentice', 'wizard', 'grandmaster'].includes(saved.difficulty)) difficulty = saved.difficulty;
    if (Object.hasOwn(pieceColors, saved.pieceColor)) pieceColor = saved.pieceColor;
    if (Object.hasOwn(pieceStyles, saved.pieceStyle)) pieceStyle = saved.pieceStyle;
    if (Object.hasOwn(backgrounds, saved.background)) background = saved.background;
  }
} catch { localStorageSafeRemove(); }
function localStorageSafeRemove() { try { localStorage.removeItem(storageKey); } catch {} }
function save() {
  try { localStorage.setItem(storageKey, JSON.stringify({ history: game.history(), difficulty, pieceColor, pieceStyle, background })); }
  catch { $('status-text').textContent = 'Browser storage is unavailable. This match will not survive a reload.'; }
}
function report(title: string, text: string) { $('status-title').textContent = title; $('status-text').textContent = text; }
function playSound(move: Move) {
  try {
    if (move.captured) audio.capture();
    else audio.move();
    if (game.isCheckmate()) audio.checkmate();
  }
  catch { report('Sound is unavailable.', 'You can continue your match without sound.'); }
}
function highlights() {
  const last = game.history({ verbose: true }).filter(move => move.san !== '--').at(-1);
  scene?.highlight(selected, selected ? game.moves({ square: selected, verbose: true }).map(move => move.to) : [], last ? [last.from, last.to] : []);
}
function notation(san: string) { return san === '--' ? '<i>skips</i>' : san; }
function render(move?: Move, sync = true) {
  if (sync) scene?.sync(game, move);
  highlights();
  const history = game.history();
  $('move-count').textContent = `${history.length} ${history.length === 1 ? 'MOVE' : 'MOVES'}`;
  if (history.length) {
    $('history').innerHTML = Array.from({ length: Math.ceil(history.length / 2) }, (_, i) => `<div class="history-row"><span>${String(i + 1).padStart(2, '0')}</span><span>${notation(history[i * 2])}</span><span>${history[i * 2 + 1] ? notation(history[i * 2 + 1]) : '<i>···</i>'}</span></div>`).join('');
    $('history').scrollTop = $('history').scrollHeight;
  } else $('history').innerHTML = '<div class="empty-history"><span>♙</span><p>Every great story<br>begins with a bold move.</p><small>Your first chapter awaits.</small></div>';
  $<HTMLButtonElement>('undo').disabled = history.length === 0;
  $<HTMLInputElement>('move-input').disabled = thinking || game.turn() === 'b' || game.isGameOver();
  let title = 'The board is yours.';
  let text = 'Select one of your pieces to see its possibilities.';
  let turn = 'Your move, wizard.';
  const skipped = history.at(-1) === '--';
  if (skipped) {
    title = game.turn() === 'w' ? 'The Guardian skips its turn.' : 'You skip this turn.';
    text = game.turn() === 'w' ? 'It has no legal move and is not in check, so you move again.' : 'You have no legal move and are not in check, so the Guardian moves again.';
  }
  if (game.isCheckmate()) {
    title = game.turn() === 'b' ? 'A magical victory.' : 'The Guardian prevails.';
    text = 'Checkmate. A fine chapter comes to a close.';
    turn = 'Checkmate';
  } else if (game.isDraw()) {
    title = 'An evenly matched duel.';
    text = game.isStalemate() ? 'Neither side has a legal move.' : game.isThreefoldRepetition() ? 'Draw by threefold repetition.' : game.isInsufficientMaterial() ? 'Draw: insufficient material to checkmate.' : 'Draw by the fifty-move rule.';
    turn = 'Match drawn';
  } else if (thinking) {
    title = 'An ancient mind at work.';
    text = skipped ? 'You have no legal move and are not in check, so the Guardian moves again.' : game.isCheck() ? 'The Guardian is finding a way out of check.' : 'The Guardian is considering its next move…';
    turn = 'The Guardian is thinking…';
  } else if (game.isCheck()) {
    title = 'Your king needs you.';
    text = 'You are in check. Move your king, block, or capture.';
    turn = 'Check — protect your king.';
  }
  const result = $('match-result');
  const wasHidden = result.hidden;
  $('result-title').textContent = game.isCheckmate() ? 'Checkmate' : 'Draw';
  $('result-winner').textContent = game.isCheckmate() ? game.turn() === 'b' ? 'You win. The Castle Guardian is defeated.' : 'The Castle Guardian wins. Try another duel.' : `${text} No one wins this duel.`;
  result.hidden = !game.isGameOver();
  if (wasHidden && !result.hidden) $('result-new-game').focus({ preventScroll: true });
  report(title, text);
  $('board-turn').innerHTML = `<span class="live-dot ${thinking ? 'thinking' : ''}"></span>${turn}`;
  save();
}
function cancelCPU() { clearTimeout(cpuTimer); cpuTimer = undefined; worker?.terminate(); worker = null; thinking = false; }
function requestCPU() {
  if (game.turn() !== 'b' || game.isGameOver()) return;
  thinking = true;
  render(undefined, false);
  try {
    worker = new Worker(new URL('./cpu.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data }: MessageEvent<{ move?: SearchReply; error?: string }>) => {
      cancelCPU();
      if (data.error || !data.move) { render(); report('The Guardian lost its train of thought.', data.error || 'Take back the turn and try again.'); return; }
      try {
        const move = game.move(data.move);
        const skipped = skipStuckTurn(game);
        selected = null;
        render(move);
        playSound(move);
        if (skipped) cpuTimer = window.setTimeout(() => { if (!thinking) requestCPU(); }, 560);
      } catch { render(); report('The Guardian could not move.', 'Take back the turn and try again.'); }
    };
    worker.onerror = () => { cancelCPU(); render(); report('The Guardian is unavailable.', 'Take back the turn or start a new game.'); };
    worker.postMessage({ history: game.history(), difficulty });
  } catch { cancelCPU(); render(); report('The Guardian is unavailable.', 'This browser could not start the chess worker.'); }
}
function makeMove(input: string | { from: string; to: string; promotion?: string }) {
  if (thinking || game.turn() !== 'w' || game.isGameOver()) return;
  try {
    const move = game.move(input);
    skipStuckTurn(game);
    selected = null;
    render(move);
    playSound(move);
    cpuTimer = window.setTimeout(() => { if (!thinking) requestCPU(); }, 560);
  } catch { report('That spell does not quite work.', 'Choose a highlighted square or enter a legal move such as e2e4.'); }
}
function selectSquare(square: Square) {
  if (thinking || game.turn() !== 'w' || game.isGameOver()) return;
  if (selected === square) { selected = null; highlights(); return; }
  if (selected) {
    const moves = game.moves({ square: selected, verbose: true }).filter(move => move.to === square);
    if (moves.length) {
      if (moves[0].promotion) {
        promotion = { from: selected, to: square };
        $<HTMLDialogElement>('promotion').showModal();
      } else makeMove({ from: selected, to: square });
      return;
    }
  }
  selected = game.get(square)?.color === 'w' ? square : null;
  highlights();
  if (selected) report(`${({ p: 'Pawn', n: 'Knight', b: 'Bishop', r: 'Rook', q: 'Queen', k: 'King' })[game.get(selected)!.type]} on ${selected}.`, 'Choose a golden ring to make your move.');
}
try { scene = new ChessScene($('scene'), selectSquare); }
catch {
  $('scene').insertAdjacentHTML('afterbegin', '<div class="webgl-fallback"><h2>The hall needs WebGL.</h2><p>Enable hardware acceleration to explore the 3D board. You can still play using the Move field below.</p><div id="fallback-board"></div></div>');
}
function updateFallback() {
  if (scene) return;
  const board = $('fallback-board');
  if (!board) return;
  const symbols = { w: { k: '♔', q: '♕', r: '♖', b: '♗', n: '♘', p: '♙' }, b: { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' } };
  board.innerHTML = game.board().flatMap((row, r) => row.map((p, f) => `<button style="color:${p ? pieceColors[p.color === 'w' ? pieceColor : pieceColor === 'white' ? 'green' : 'white'] : 'inherit'}" data-square="${String.fromCharCode(97 + f)}${8 - r}" aria-label="${String.fromCharCode(97 + f)}${8 - r}${p ? ` ${p.color} ${p.type}` : ''}">${p ? symbols[p.color][p.type] : ''}</button>`)).join('');
  board.querySelectorAll<HTMLButtonElement>('button').forEach(button => { button.onclick = () => { selectSquare(button.dataset.square as Square); updateFallback(); }; });
}
if (!scene) new MutationObserver(updateFallback).observe($('history'), { childList: true });
$('move-form').addEventListener('submit', event => {
  event.preventDefault();
  const input = $<HTMLInputElement>('move-input');
  const value = input.value.trim();
  if (!value) return;
  if (/^[a-h][1-8][a-h][1-8][qrbn]?$/i.test(value)) {
    const v = value.toLowerCase();
    const from = v.slice(0, 2) as Square;
    const to = v.slice(2, 4) as Square;
    if (v.length === 4 && game.moves({ square: from, verbose: true }).some(move => move.to === to && move.promotion)) {
      promotion = { from, to };
      $<HTMLDialogElement>('promotion').showModal();
    } else makeMove({ from, to, promotion: v[4] });
  } else makeMove(value);
  input.value = '';
});
$('undo').onclick = () => {
  cancelCPU();
  if (game.history().length) {
    let undone;
    do undone = game.undo();
    while (undone && (undone.color === 'b' || undone.san === '--'));
  }
  selected = null;
  render();
};
function restart() {
  cancelCPU();
  game.reset();
  selected = null;
  promotion = null;
  render();
  $<HTMLDialogElement>('restart').close();
}
$('new-game').onclick = () => game.history().length ? $<HTMLDialogElement>('restart').showModal() : restart();
$('confirm-restart').onclick = restart;
$('result-new-game').onclick = restart;
$('cancel-restart').onclick = () => $<HTMLDialogElement>('restart').close();
$('guide-button').onclick = () => $<HTMLDialogElement>('guide').showModal();
for (const button of document.querySelectorAll<HTMLButtonElement>('.dialog-close, .dialog-done')) button.onclick = () => $<HTMLDialogElement>('guide').close();
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-piece]')) button.onclick = () => {
  if (promotion) makeMove({ ...promotion, promotion: button.dataset.piece });
  promotion = null;
  $<HTMLDialogElement>('promotion').close();
};
$('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $('game-layout').requestFullscreen();
  } catch { report('Fullscreen is unavailable.', 'Your browser could not enter fullscreen. You can keep playing here.'); }
};
document.addEventListener('fullscreenchange', () => {
  const fullscreen = document.fullscreenElement === $('game-layout');
  const label = fullscreen ? 'Exit game fullscreen' : 'Enter game fullscreen';
  $('fullscreen').setAttribute('aria-label', label);
  $('fullscreen').setAttribute('title', label);
  $('fullscreen').setAttribute('aria-pressed', String(fullscreen));
  $('fullscreen').classList.toggle('active', fullscreen);
});
if (!document.fullscreenEnabled) {
  $<HTMLButtonElement>('fullscreen').disabled = true;
  $('fullscreen').setAttribute('title', 'Fullscreen is unavailable in this browser');
}
$('flip').onclick = () => scene?.flip();
$('view').onclick = () => {
  const overhead = scene?.toggleView();
  $('view').setAttribute('aria-label', overhead ? 'Switch to perspective view' : 'Switch to overhead view');
  $('view').setAttribute('title', overhead ? 'Switch to perspective view' : 'Switch to overhead view');
  $('view').classList.toggle('active', Boolean(overhead));
};
function updateSoundControl() {
  $('sound').setAttribute('aria-pressed', String(sound));
  const label = sound ? 'Mute library music and sound' : 'Play library music and sound';
  $('sound').setAttribute('aria-label', label);
  $('sound').setAttribute('title', label);
  $('sound').classList.toggle('active', sound);
  $('music-status').textContent = sound ? 'HEDWIG’S THEME · NOW PLAYING' : 'HEDWIG’S THEME · SOUND OFF';
  musicReady = false;
  $('music-panel').hidden = !sound;
  $('music-panel').classList.remove('playing');
  $<HTMLIFrameElement>('music-player').src = sound ? music : 'about:blank';
}
let musicReady = false;
$('music-player').addEventListener('load', () => {
  musicReady = sound;
  if (sound) $('sound').focus();
});
window.addEventListener('blur', () => {
  if (musicReady && document.activeElement === $('music-player')) $('music-panel').classList.add('playing');
});
$('sound').onclick = async () => {
  sound = !sound;
  updateSoundControl();
  try { await audio.setEnabled(sound); }
  catch {
    sound = false;
    await audio.setEnabled(false);
    updateSoundControl();
    report('Music could not start.', 'Tap the music control to try again.');
  }
  updateFire();
};
function updateFire() { audio.fire(sound && background === 'library'); }
function updateDifficulty() {
  $<HTMLSelectElement>('difficulty').value = difficulty;
  $('difficulty-note').textContent = { apprentice: 'A gentler match. One move at a time.', wizard: 'A thoughtful duel. Two plies deep.', grandmaster: 'A sharper mind. Up to three plies deep.' }[difficulty];
}
$('difficulty').onchange = () => {
  difficulty = $<HTMLSelectElement>('difficulty').value as Difficulty;
  updateDifficulty();
  save();
  if (thinking) { cancelCPU(); requestCPU(); }
};
function updatePieceColor() {
  scene?.setPieceColor(pieceColor);
  const cpuColor = pieceColor === 'white' ? 'green' : 'white';
  $('human-heading').textContent = pieceColor.toUpperCase();
  $('cpu-heading').textContent = cpuColor.toUpperCase();
  $('cpu-color').textContent = cpuColor[0].toUpperCase() + cpuColor.slice(1);
  document.querySelectorAll<HTMLInputElement>('[name="piece-color"]').forEach(input => { input.checked = input.value === pieceColor; });
  updateFallback();
}
for (const input of document.querySelectorAll<HTMLInputElement>('[name="piece-color"]')) input.onchange = () => {
  if (!input.checked) return;
  pieceColor = input.value as PieceColor;
  updatePieceColor();
  save();
};
function updateLook() {
  scene?.setPieceStyle(pieceStyle);
  scene?.setBackground(background);
  $<HTMLSelectElement>('piece-style').value = pieceStyle;
  $<HTMLSelectElement>('background').value = background;
  $('room-name').textContent = rooms[background].label;
  $('room-caption').textContent = rooms[background].caption;
  updateFire();
}
$('piece-style').onchange = () => {
  pieceStyle = $<HTMLSelectElement>('piece-style').value as PieceStyle;
  updateLook();
  save();
};
$('background').onchange = () => {
  background = $<HTMLSelectElement>('background').value as Background;
  updateLook();
  save();
};
updatePieceColor();
updateLook();
updateDifficulty();
render();
updateFallback();
requestCPU();
