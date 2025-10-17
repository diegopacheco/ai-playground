export function createGroupingToggle(isGrouped, onToggle) {
  const button = document.createElement('button');
  button.className = 'btn btn-secondary grouping-toggle';
  button.setAttribute('data-testid', 'grouping-toggle');
  button.setAttribute('aria-label', isGrouped ? 'Switch to flat view' : 'Group by date');
  button.setAttribute('aria-pressed', isGrouped.toString());

  const icon = document.createElement('span');
  icon.className = 'toggle-icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.textContent = isGrouped ? '▤' : '☰';

  const label = document.createElement('span');
  label.className = 'toggle-label';
  label.textContent = isGrouped ? 'Flat View' : 'Group by Date';

  button.appendChild(icon);
  button.appendChild(label);

  button.addEventListener('click', () => {
    const newGroupedState = !isGrouped;
    localStorage.setItem('album-grouping-enabled', newGroupedState.toString());
    onToggle(newGroupedState);
  });

  return button;
}

export function getGroupingPreference() {
  const stored = localStorage.getItem('album-grouping-enabled');
  return stored === 'true';
}
