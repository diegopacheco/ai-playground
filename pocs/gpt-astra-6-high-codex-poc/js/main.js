import { spots, createGame, startGame, jump, performTrick, togglePause, updateGame } from './game.js';
import { drawScene, drawPreview } from './renderer.js';

const $ = id => document.getElementById(id);
const canvas = $('game');
const ctx = canvas.getContext('2d');
let game = createGame();
let best = 0;
try { best = Math.max(0, Number(localStorage.getItem('sidewalk-best')) || 0); } catch {}
let previous = performance.now();
let overlayStatus = '';
let guideWasPlaying = false;
const formatScore = value => Math.floor(value).toLocaleString('en-US');
function saveBest() {
  if (game.score <= best) return;
  best = game.score;
  try { localStorage.setItem('sidewalk-best', String(best)); } catch {}
}
function reset(spot = game.spot, rider = game.rider) {
  saveBest();
  game = createGame(spot, rider);
  $('spot-label').textContent = spot.name.toUpperCase();
  $('spot-caption').textContent = spot.caption;
  $('scene-name').textContent = spot.title;
  $('scene-detail').textContent = spot.detail;
  document.querySelectorAll('[data-spot]').forEach(button => {
    const selected = button.dataset.spot === spot.id;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  document.querySelectorAll('[data-rider]').forEach(button => {
    const selected = button.dataset.rider === rider;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  sync();
}
function begin() {
  if (game.status === 'finished') reset();
  if (game.status === 'paused') togglePause(game);
  else startGame(game);
  canvas.focus({ preventScroll: true });
  sync();
}
function action(name) {
  if (name === 'pause') togglePause(game);
  else if (name === 'jump') jump(game);
  else performTrick(game, name);
  sync();
}
function sync() {
  $('score').textContent = String(game.score).padStart(5, '0');
  const seconds = Math.ceil(game.time);
  $('time').textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  $('best').innerHTML = `${formatScore(best)} <small>PTS</small>`;
  $('trick-toast').textContent = game.status === 'playing' && game.messageTime > 0 ? game.message : '';
  $('pause').hidden = game.status !== 'playing';
  canvas.setAttribute('data-status', game.status);
  if (overlayStatus === game.status) return;
  overlayStatus = game.status;
  $('start-overlay').hidden = game.status === 'playing';
  const content = game.status === 'finished'
    ? ['THAT’S A WRAP', 'A good day on four wheels.', `${formatScore(game.score)} points · ${game.landed} clean landings · ${game.bails} bails`, 'Ride again']
    : game.status === 'paused'
      ? ['TAKE A BREATHER', 'Your line can wait.', 'Your session is paused. Pick up right where you left off.', 'Keep rolling']
      : ['THE STREETS ARE WAITING', 'Make it a good session.', '60 seconds. Four iconic spots. Your own kind of flow.', 'Let’s skate'];
  $('overlay-kicker').textContent = content[0];
  $('overlay-title').textContent = content[1];
  $('overlay-copy').textContent = content[2];
  $('start').innerHTML = `${content[3]} <span>→</span>`;
}
$('spots').innerHTML = spots.map((spot, index) => `<button class="spot ${index === 0 ? 'selected' : ''}" data-spot="${spot.id}" aria-pressed="${index === 0}"><span class="spot-icon"><svg viewBox="0 0 38 32" aria-hidden="true"><path d="${spot.icon}"/></svg></span><span class="spot-copy"><strong>${spot.name}</strong><small>${spot.type}</small></span><span class="spot-arrow">${index === 0 ? '↗' : '→'}</span></button>`).join('');
document.querySelectorAll('[data-spot]').forEach(button => button.addEventListener('click', () => reset(spots.find(spot => spot.id === button.dataset.spot))));
document.querySelectorAll('[data-rider]').forEach(button => button.addEventListener('click', () => reset(game.spot, button.dataset.rider)));
document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => action(button.dataset.action)));
$('start').addEventListener('click', begin);
$('pause').addEventListener('click', () => action('pause'));
window.addEventListener('keydown', event => {
  if ($('guide').open || event.repeat) return;
  if (event.key === 'Enter' && (event.target === canvas || event.target === document.body)) {
    event.preventDefault();
    begin();
    return;
  }
  if (event.target instanceof HTMLButtonElement && [' ', 'Enter'].includes(event.key)) return;
  const keys = { ' ': 'jump', ArrowUp: 'jump', j: 'kickflip', k: 'heelflip', l: 'spin', p: 'pause' };
  const name = keys[event.key] || keys[event.key.toLowerCase()];
  if (name) { event.preventDefault(); action(name); }
});
$('guide-open').addEventListener('click', () => {
  guideWasPlaying = game.status === 'playing';
  if (guideWasPlaying) togglePause(game);
  $('guide').showModal();
  sync();
});
$('guide-close').addEventListener('click', () => $('guide').close());
$('guide-ready').addEventListener('click', () => $('guide').close());
$('guide').addEventListener('close', () => {
  if (guideWasPlaying && game.status === 'paused') togglePause(game);
  guideWasPlaying = false;
  sync();
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && game.status === 'playing') { togglePause(game); sync(); }
});
window.addEventListener('pagehide', saveBest);
for (const rider of ['boy', 'girl']) drawPreview($(`${rider}-preview`), rider);
function frame(now) {
  updateGame(game, (now - previous) / 1000);
  previous = now;
  saveBest();
  drawScene(ctx, game, now / 1000);
  sync();
  requestAnimationFrame(frame);
}
sync();
requestAnimationFrame(frame);
