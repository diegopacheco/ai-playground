const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

const CAUSES = { swatter: 'swatted', spider: 'eaten by the spider', age: 'old age' };

export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ESCAPES[char]);
}

export function renderBuzzHtml(text) {
  return escapeHtml(text).replace(/(^|\s)([@#]\w+)/g, (match, lead, token) => {
    const kind = token.startsWith('@') ? 'mention' : 'tag';
    return `${lead}<span class="${kind}">${token}</span>`;
  });
}

export function causeLabel(cause) {
  return CAUSES[cause] || 'unknown';
}

export function lifePercent(ageDays, lifespanDays) {
  if (!lifespanDays) return 100;
  return Math.max(0, Math.min(100, Math.round((ageDays / lifespanDays) * 100)));
}

export function avatarSvg(species) {
  const body = escapeHtml(species?.body || '#444');
  const eyes = escapeHtml(species?.eyes || '#a00');
  return `<svg class="avatar" viewBox="0 0 40 40" aria-hidden="true">
    <circle cx="20" cy="20" r="20" fill="#4a4460"/>
    <ellipse cx="11" cy="17" rx="8" ry="4" fill="#dfe9ff" opacity="0.55" transform="rotate(-30 11 17)"/>
    <ellipse cx="29" cy="17" rx="8" ry="4" fill="#dfe9ff" opacity="0.55" transform="rotate(30 29 17)"/>
    <ellipse cx="20" cy="25" rx="7" ry="9" fill="${body}"/>
    <circle cx="20" cy="15" r="6" fill="${body}"/>
    <circle cx="15.5" cy="13" r="4" fill="${eyes}"/>
    <circle cx="24.5" cy="13" r="4" fill="${eyes}"/>
    <circle cx="14.5" cy="12" r="1.2" fill="#fff" opacity="0.8"/>
    <circle cx="23.5" cy="12" r="1.2" fill="#fff" opacity="0.8"/>
  </svg>`;
}
