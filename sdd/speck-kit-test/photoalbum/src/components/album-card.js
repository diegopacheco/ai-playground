export function createAlbumCard(album, onClick, dragHandlers = null) {
  const card = document.createElement('div');
  card.className = 'album-card';
  card.setAttribute('data-testid', 'album-card');
  card.setAttribute('data-album-id', album.id);
  card.setAttribute('draggable', 'true');
  card.setAttribute('role', 'button');
  card.setAttribute('aria-label', `Album: ${album.name}. Draggable. Press space to reorder.`);

  const thumbnail = document.createElement('div');
  thumbnail.className = 'album-thumbnail';

  if (album.thumbnail_ref) {
    const img = document.createElement('img');
    img.src = album.thumbnail_ref;
    img.alt = album.name;
    thumbnail.appendChild(img);
  } else {
    thumbnail.textContent = '📷';
  }

  const info = document.createElement('div');
  info.className = 'album-info';

  const name = document.createElement('h3');
  name.className = 'album-name';
  name.textContent = album.name;

  const metadata = document.createElement('div');
  metadata.className = 'album-metadata';

  const date = document.createElement('span');
  date.className = 'album-date';
  date.setAttribute('data-testid', 'album-date');
  date.textContent = formatDate(album.created_date);

  metadata.appendChild(date);

  info.appendChild(name);
  info.appendChild(metadata);

  card.appendChild(thumbnail);
  card.appendChild(info);

  if (onClick) {
    card.addEventListener('click', () => onClick(album));
    card.style.cursor = 'pointer';
  }

  if (dragHandlers) {
    card.addEventListener('dragstart', (e) => dragHandlers.onDragStart(e, card, album));
    card.addEventListener('dragend', (e) => dragHandlers.onDragEnd(e, card));
    card.addEventListener('dragover', (e) => dragHandlers.onDragOver(e, card));
    card.addEventListener('dragenter', (e) => dragHandlers.onDragEnter(e, card));
    card.addEventListener('dragleave', (e) => dragHandlers.onDragLeave(e, card));
    card.addEventListener('drop', (e) => dragHandlers.onDrop(e, card));
  }

  return card;
}

function formatDate(dateString) {
  if (!dateString) return '';

  try {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  } catch {
    return '';
  }
}
