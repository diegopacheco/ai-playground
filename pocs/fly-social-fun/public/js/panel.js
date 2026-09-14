import { avatarSvg, causeLabel, escapeHtml, lifePercent, renderBuzzHtml } from './format.js';

const MAX_FEED_ITEMS = 80;

export function createPanel(root, { onSelectFly }) {
  const statsEl = root.querySelector('#stats');
  const feedEl = root.querySelector('#tab-feed');
  const trendingEl = root.querySelector('#tab-trending');
  const topEl = root.querySelector('#tab-top');
  let species = {};

  root.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      root.querySelectorAll('.tab').forEach((other) => other.classList.toggle('active', other === tab));
      root.querySelectorAll('.tab-body').forEach((body) => (body.hidden = body.id !== `tab-${tab.dataset.tab}`));
    });
  });

  root.addEventListener('click', (event) => {
    const target = event.target.closest('[data-fly]');
    if (target) onSelectFly(target.dataset.fly);
  });

  function buzzHtml(buzz) {
    const reply = buzz.replyTo ? '<div class="reply-flag">replying in thread</div>' : '';
    return `<article class="buzz" data-fly="${escapeHtml(buzz.flyId)}">
      ${avatarSvg(species[buzz.species])}
      <div class="buzz-body">
        <header><strong>${escapeHtml(buzz.name)}</strong><span class="handle">@${escapeHtml(buzz.handle)}</span><span class="when">day ${buzz.day}</span></header>
        ${reply}
        <p>${renderBuzzHtml(buzz.text)}</p>
        <footer><span class="likes" data-buzz="${escapeHtml(buzz.id)}">&#9829; <b>${buzz.likes}</b></span></footer>
      </div>
    </article>`;
  }

  function reset(snapshot) {
    species = snapshot.species;
    feedEl.innerHTML = snapshot.feed.length ? snapshot.feed.map(buzzHtml).join('') : '<p class="empty">The flies are waking up...</p>';
    update(snapshot);
  }

  function addBuzz(buzz) {
    feedEl.querySelector('.empty')?.remove();
    feedEl.insertAdjacentHTML('afterbegin', buzzHtml(buzz));
    feedEl.firstElementChild.classList.add('fresh');
    while (feedEl.children.length > MAX_FEED_ITEMS) feedEl.lastElementChild.remove();
  }

  function updateLikes(buzzId, likes) {
    const el = feedEl.querySelector(`.likes[data-buzz="${CSS.escape(buzzId)}"]`);
    if (!el) return;
    el.querySelector('b').textContent = likes;
    el.classList.remove('pop');
    void el.offsetWidth;
    el.classList.add('pop');
  }

  function renderStats(stats) {
    const items = [['alive', stats.alive], ['buzzes', stats.buzzes], ['hatched', stats.hatched], ['RIP', stats.deaths], ['day', stats.day]];
    statsEl.innerHTML = items.map(([label, value]) => `<div><b>${value}</b><span>${label}</span></div>`).join('');
  }

  function renderTrending(trending, memoriam) {
    const max = Math.max(1, ...trending.map((item) => item.count));
    const tags = trending.length
      ? trending.map((item, index) => `<li><span class="rank">${index + 1}</span><div class="trend"><b class="tag">${escapeHtml(item.tag)}</b><div class="bar"><i style="width:${(item.count / max) * 100}%"></i></div></div><span class="count">${item.count} buzzes</span></li>`).join('')
      : '<li class="empty">Nothing trending yet</li>';
    const graves = memoriam.length
      ? memoriam.map((fly) => `<li data-fly="${escapeHtml(fly.id)}">${avatarSvg(species[fly.species])}<div><b>${escapeHtml(fly.name)}</b><span>${escapeHtml(causeLabel(fly.cause))} at ${fly.ageDays} days</span></div></li>`).join('')
      : '<li class="empty">Everyone is still buzzing</li>';
    trendingEl.innerHTML = `<h2>Trending in the kitchen</h2><ol class="trending">${tags}</ol><h2>In memoriam</h2><ul class="memoriam">${graves}</ul>`;
  }

  function renderTop(top) {
    const rows = top.map((fly, index) => `<li data-fly="${escapeHtml(fly.id)}">
      <span class="rank">${index + 1}</span>
      ${avatarSvg(species[fly.species])}
      <div class="who"><b>${escapeHtml(fly.name)}</b><span>@${escapeHtml(fly.handle)} - ${escapeHtml(fly.speciesLabel)}</span>
        <div class="life" title="life used"><i style="width:${lifePercent(fly.ageDays, fly.lifespanDays)}%"></i></div></div>
      <div class="numbers"><b>${fly.followers}</b><span>followers</span></div>
      <div class="numbers"><b>${fly.likes}</b><span>likes</span></div>
    </li>`);
    topEl.innerHTML = `<h2>Most followed flies</h2><ol class="top">${rows.join('')}</ol>`;
  }

  function update({ stats, trending, top, memoriam }) {
    renderStats(stats);
    renderTrending(trending, memoriam);
    renderTop(top);
  }

  return { reset, addBuzz, updateLikes, update };
}
