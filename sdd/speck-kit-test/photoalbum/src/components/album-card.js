const ALBUM_PLACEHOLDER = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"%3E%3Crect fill="%23f1f5f9" width="400" height="300"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dominant-baseline="middle" font-size="64" fill="%2394a3b8"%3E📷%3C/text%3E%3C/svg%3E';

export function createAlbumCard(album, onClick, dragHandlers = null, onDelete = null) {
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
    img.alt = album.name;
    img.decoding = 'async';
    img.loading = 'lazy';
    img.className = 'album-thumbnail-img';

    img.onload = () => {
      img.classList.add('loaded');
    };

    img.onerror = () => {
      img.src = ALBUM_PLACEHOLDER;
      img.classList.add('error');
    };

    img.src = album.thumbnail_ref;
    thumbnail.appendChild(img);
  } else {
    const placeholder = document.createElement('img');
    placeholder.src = ALBUM_PLACEHOLDER;
    placeholder.alt = album.name;
    placeholder.className = 'album-thumbnail-img loaded';
    thumbnail.appendChild(placeholder);
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

  if (onDelete) {
    const deleteButton = document.createElement('button');
    deleteButton.className = 'btn-icon btn-delete';
    deleteButton.setAttribute('data-testid', 'delete-album-btn');
    deleteButton.setAttribute('aria-label', `Delete ${album.name}`);
    deleteButton.textContent = '🗑️';
    deleteButton.addEventListener('click', (e) => {
      e.stopPropagation();
      onDelete(album);
    });
    metadata.appendChild(deleteButton);
  }

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
