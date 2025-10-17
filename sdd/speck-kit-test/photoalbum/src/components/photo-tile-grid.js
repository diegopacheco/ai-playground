export function createPhotoTileGrid(photos, onPhotoClick) {
  const grid = document.createElement('div');
  grid.className = 'photo-grid';
  grid.setAttribute('data-testid', 'photo-grid');

  if (photos.length === 0) {
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.setAttribute('data-testid', 'empty-state');
    emptyState.innerHTML = `
      <p>No photos in this album yet. Click "Add Photos" to get started!</p>
    `;
    grid.appendChild(emptyState);
    return grid;
  }

  photos.forEach(photo => {
    const tile = createPhotoTile(photo, onPhotoClick);
    grid.appendChild(tile);
  });

  return grid;
}

function createPhotoTile(photo, onClick) {
  const tile = document.createElement('div');
  tile.className = 'photo-tile';
  tile.setAttribute('data-testid', 'photo-tile');
  tile.setAttribute('data-photo-id', photo.photo_id || photo.id);

  const thumbnail = document.createElement('img');
  thumbnail.className = 'photo-thumbnail';
  thumbnail.setAttribute('data-testid', 'photo-thumbnail');
  thumbnail.src = photo.thumbnail_blob || 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg"%3E%3C/svg%3E';
  thumbnail.alt = photo.file_path || 'Photo';
  thumbnail.loading = 'lazy';

  const metadata = document.createElement('div');
  metadata.className = 'photo-metadata';
  metadata.setAttribute('data-testid', 'photo-metadata');
  metadata.style.display = 'none';

  if (photo.date_taken) {
    const date = document.createElement('div');
    date.className = 'photo-date';
    date.setAttribute('data-testid', 'photo-date');
    date.textContent = formatDate(photo.date_taken);
    metadata.appendChild(date);
  }

  if (photo.width && photo.height) {
    const dimensions = document.createElement('div');
    dimensions.className = 'photo-dimensions';
    dimensions.textContent = `${photo.width} × ${photo.height}`;
    metadata.appendChild(dimensions);
  }

  tile.appendChild(thumbnail);
  tile.appendChild(metadata);

  tile.addEventListener('mouseenter', () => {
    metadata.style.display = 'block';
  });

  tile.addEventListener('mouseleave', () => {
    metadata.style.display = 'none';
  });

  if (onClick) {
    tile.addEventListener('click', () => onClick(photo));
    tile.style.cursor = 'pointer';
  }

  return tile;
}

function formatDate(dateString) {
  if (!dateString) return '';

  try {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch {
    return '';
  }
}
