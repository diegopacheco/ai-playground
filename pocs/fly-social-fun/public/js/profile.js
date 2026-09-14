import { avatarSvg, causeLabel, escapeHtml, lifePercent, renderBuzzHtml } from './format.js';

export function createProfile(el, { load, spotLabel, speciesOf, onClose }) {
  let openId = null;

  el.addEventListener('click', (event) => {
    if (event.target.closest('.close')) close();
  });

  function close() {
    openId = null;
    el.hidden = true;
    onClose();
  }

  function render(fly) {
    const status = fly.alive
      ? `<span class="status alive">buzzing around the ${escapeHtml(spotLabel(fly.spot))}</span>`
      : `<span class="status dead">RIP - ${escapeHtml(causeLabel(fly.cause))}</span>`;
    const parents = fly.parents.length ? `<p class="parents">child of ${fly.parents.map((handle) => `@${escapeHtml(handle)}`).join(' &amp; ')}</p>` : '';
    const buzzes = fly.buzzes.length
      ? fly.buzzes.map((buzz) => `<li><span class="when">day ${buzz.day}</span>${renderBuzzHtml(buzz.text)}</li>`).join('')
      : '<li class="empty">No buzzes yet. Shy fly.</li>';
    el.innerHTML = `
      <button class="close" aria-label="close">&times;</button>
      <div class="profile-head">${avatarSvg(speciesOf(fly.species))}<div><h3>${escapeHtml(fly.name)}</h3><span class="handle">@${escapeHtml(fly.handle)}</span></div></div>
      <span class="chip">${escapeHtml(fly.speciesLabel)}</span> ${status}
      <p class="bio">${escapeHtml(fly.bio)}</p>
      ${parents}
      <div class="profile-stats">
        <div><b>${fly.followers}</b><span>followers</span></div>
        <div><b>${fly.following}</b><span>following</span></div>
        <div><b>${fly.buzzCount}</b><span>buzzes</span></div>
        <div><b>${fly.likes}</b><span>likes</span></div>
      </div>
      <label>life: day ${fly.ageDays} of ${fly.lifespanDays}</label>
      <div class="life"><i style="width:${lifePercent(fly.ageDays, fly.lifespanDays)}%"></i></div>
      <label>hunger</label>
      <div class="life hunger"><i style="width:${Math.round(fly.hunger * 100)}%"></i></div>
      <h4>Latest buzzes</h4>
      <ul class="profile-buzzes">${buzzes}</ul>`;
  }

  async function refresh() {
    if (!openId) return;
    const id = openId;
    const fly = await load(id);
    if (id !== openId) return;
    if (!fly) return close();
    render(fly);
    el.hidden = false;
  }

  function open(flyId) {
    openId = flyId;
    return refresh();
  }

  return { open, refresh, close };
}
