import { formatTime, nextIndex, previousIndex, removeTrack, scrollText, totalSeconds } from './playlist.js';
import { createPlayer, ENDED, PLAYING, PAUSED, BUFFERING } from './player.js';
import { createVisualizer, MODES } from './visualizer.js';

const STORE = 'younap';
const TITLE_WIDTH = 34;
const IDLE_TITLE = 'YOUNAP - SEARCH YOUTUBE AND DOUBLE CLICK A SONG';
const OPENING = {
  id: 'bBY52n7luM8',
  title: "Winamp Startup Stinger - It really whips the llama's ass",
  channel: 'Casually',
  duration: '0:06',
  seconds: 6
};
const LABELS = { [PLAYING]: 'PLAY', [PAUSED]: 'PAUSE', [BUFFERING]: 'BUFFER' };

const $ = id => document.getElementById(id);

const state = { tracks: [OPENING], shuffle: false, repeat: false, volume: 80, visual: 'BARS', ...load(), current: -1, selected: -1 };

let player = null;
let playing = false;
let status = 'STOP';
let seeking = false;
let failures = 0;
let ticks = 0;
let wanted = -1;
let results = [];

function load() {
  try {
    return JSON.parse(localStorage.getItem(STORE)) || {};
  } catch {
    return {};
  }
}

function save() {
  const { tracks, shuffle, repeat, volume, visual } = state;
  try {
    localStorage.setItem(STORE, JSON.stringify({ tracks, shuffle, repeat, volume, visual }));
  } catch {
    return;
  }
}

function row(track, index, classes) {
  const item = document.createElement('li');
  item.dataset.index = index;
  item.className = classes;
  const title = document.createElement('span');
  title.className = 'row-title';
  title.textContent = `${index + 1}. ${track.title} - ${track.channel}`;
  const duration = document.createElement('span');
  duration.textContent = track.duration;
  item.append(title, duration);
  return item;
}

function renderPlaylist() {
  const rows = state.tracks.map((track, index) => {
    const classes = [
      index === state.current ? 'current' : '',
      index === state.selected ? 'selected' : '',
      track.blocked ? 'blocked' : ''
    ];
    return row(track, index, classes.join(' ').trim());
  });
  $('playlist').replaceChildren(...rows);
  $('summary').textContent = `${state.tracks.length} TRACKS / ${formatTime(totalSeconds(state.tracks))}`;
  const shown = state.current < 0 ? 0 : state.current + 1;
  $('position').textContent = `${String(shown).padStart(2, '0')}/${String(state.tracks.length).padStart(2, '0')}`;
}

function renderResults() {
  const rows = results.map((track, index) => {
    const item = row(track, index, '');
    const add = document.createElement('button');
    add.className = 'add';
    add.textContent = '+';
    add.title = 'Enqueue';
    item.append(add);
    return item;
  });
  $('results').replaceChildren(...rows);
}

function renderModes() {
  $('shuffle').setAttribute('aria-pressed', state.shuffle);
  $('repeat').setAttribute('aria-pressed', state.repeat);
  const buttons = Object.keys(MODES).map(mode => {
    const button = document.createElement('button');
    button.className = 'btn text toggle';
    button.textContent = mode;
    button.dataset.mode = mode;
    button.setAttribute('aria-pressed', mode === state.visual);
    return button;
  });
  $('modes').replaceChildren(...buttons);
}

function renderTitle() {
  const track = state.tracks[state.current];
  const text = track ? `${state.current + 1}. ${track.title} - ${track.channel} (${track.duration})` : IDLE_TITLE;
  $('title').textContent = scrollText(text.toUpperCase(), ticks, TITLE_WIDTH);
}

function renderClock() {
  const elapsed = status === 'STOP' ? 0 : player.getCurrentTime();
  const duration = player.getDuration() || 0;
  $('clock').textContent = formatTime(elapsed);
  if (!seeking) $('seek').value = duration ? Math.round((elapsed / duration) * 1000) : 0;
}

function play(index) {
  if (!player) {
    wanted = index;
    return;
  }
  if (index < 0 || index >= state.tracks.length) return stop();
  state.current = index;
  state.selected = index;
  ticks = 0;
  player.loadVideoById(state.tracks[index].id);
  renderPlaylist();
}

function resume() {
  if (!player || state.current < 0) return play(Math.max(state.selected, 0));
  player.playVideo();
}

function pause() {
  if (player) player.pauseVideo();
}

function stop() {
  if (!player) return;
  player.stopVideo();
  setStatus('STOP');
}

function setStatus(label) {
  status = label;
  playing = label === 'PLAY';
  $('status').textContent = label;
}

function next() {
  play(nextIndex(state.current, state.tracks.length, state));
}

function previous() {
  play(previousIndex(state.current, state.tracks.length));
}

function onState(code) {
  if (code === ENDED) return next();
  setStatus(LABELS[code] || 'STOP');
  if (playing) failures = 0;
}

function onError() {
  const track = state.tracks[state.current];
  if (track) track.blocked = true;
  failures++;
  save();
  if (failures < state.tracks.length) next();
  else stop();
  renderPlaylist();
}

function enqueue(...tracks) {
  state.tracks.push(...tracks.map(track => ({ ...track })));
  save();
  renderPlaylist();
}

function remove(index) {
  const wasCurrent = index === state.current;
  const { tracks, current } = removeTrack(state.tracks, state.current, index);
  state.tracks = tracks;
  state.current = current;
  state.selected = Math.min(state.selected, tracks.length - 1);
  if (wasCurrent) stop();
  save();
  renderPlaylist();
}

function clear() {
  state.tracks = [];
  state.current = -1;
  state.selected = -1;
  stop();
  save();
  renderPlaylist();
}

async function runSearch(event) {
  event.preventDefault();
  const query = $('query').value.trim();
  if (!query) return;
  $('search-status').textContent = 'SEARCHING...';
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error);
    results = body;
    $('search-status').textContent = `${results.length} RESULTS - DOUBLE CLICK PLAYS, + ENQUEUES`;
  } catch (error) {
    results = [];
    $('search-status').textContent = `SEARCH FAILED: ${error.message}`.toUpperCase();
  }
  renderResults();
}

function indexOf(event) {
  const item = event.target.closest('li');
  return item ? Number(item.dataset.index) : -1;
}

function toggle(mode) {
  state[mode] = !state[mode];
  save();
  renderModes();
}

const KEYS = { z: previous, x: resume, c: pause, v: stop, b: next };

function onKey(event) {
  if (event.target.tagName === 'INPUT' || event.metaKey || event.ctrlKey) return;
  if (KEYS[event.key]) KEYS[event.key]();
  if (event.key === 'Delete' || event.key === 'Backspace') remove(state.selected);
}

function bind() {
  $('previous').onclick = previous;
  $('play').onclick = resume;
  $('pause').onclick = pause;
  $('stop').onclick = stop;
  $('next').onclick = next;
  $('eject').onclick = () => $('query').focus();
  $('shuffle').onclick = () => toggle('shuffle');
  $('repeat').onclick = () => toggle('repeat');
  $('remove').onclick = () => remove(state.selected);
  $('clear').onclick = clear;
  $('add-all').onclick = () => enqueue(...results);
  $('modes').onclick = event => {
    if (!event.target.dataset.mode) return;
    state.visual = event.target.dataset.mode;
    save();
    renderModes();
  };
  $('stage').onclick = () => {
    const names = Object.keys(MODES);
    state.visual = names[(names.indexOf(state.visual) + 1) % names.length];
    save();
    renderModes();
  };
  $('search').onsubmit = runSearch;
  $('volume').value = state.volume;
  $('volume').oninput = event => {
    state.volume = Number(event.target.value);
    if (player) player.setVolume(state.volume);
    save();
  };
  $('seek').oninput = () => {
    seeking = true;
  };
  $('seek').onchange = event => {
    seeking = false;
    if (player && state.current >= 0) player.seekTo((event.target.value / 1000) * player.getDuration(), true);
  };
  $('playlist').onclick = event => {
    state.selected = indexOf(event);
    [...$('playlist').children].forEach((item, index) => item.classList.toggle('selected', index === state.selected));
  };
  $('playlist').ondblclick = event => {
    const index = indexOf(event);
    if (index >= 0) play(index);
  };
  $('results').onclick = event => {
    if (event.target.className === 'add') enqueue(results[indexOf(event)]);
  };
  $('results').ondblclick = event => {
    const index = indexOf(event);
    if (index < 0 || event.target.className === 'add') return;
    enqueue(results[index]);
    play(state.tracks.length - 1);
  };
  document.addEventListener('keydown', onKey);
}

async function start() {
  bind();
  renderModes();
  renderPlaylist();
  renderTitle();
  createVisualizer($('visualizer'), () => playing, () => 'BARS');
  createVisualizer($('stage'), () => playing, () => state.visual);
  setInterval(() => {
    ticks++;
    renderTitle();
    if (player) renderClock();
  }, 250);
  if (state.tracks[0]?.id === OPENING.id) {
    play(0);
    document.addEventListener('pointerdown', () => {
      if (status === 'STOP' && state.current === 0) resume();
    }, { once: true });
  }
  player = await createPlayer('video', { onState, onError });
  player.setVolume(state.volume);
  if (wanted >= 0) play(wanted);
}

start();
