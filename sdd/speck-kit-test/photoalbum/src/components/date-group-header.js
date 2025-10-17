import { formatDateGroup } from '../utils/date-grouping.js';

export function createDateGroupHeader(dateGroup) {
  const header = document.createElement('div');
  header.className = 'date-group-header';
  header.setAttribute('data-testid', 'date-group-header');
  header.setAttribute('role', 'heading');
  header.setAttribute('aria-level', '2');

  const title = document.createElement('h2');
  title.className = 'date-group-title';
  title.textContent = formatDateGroup(dateGroup);

  header.appendChild(title);

  return header;
}
