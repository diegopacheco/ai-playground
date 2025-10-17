export function createPhotoTile(photo, onClick) {
  const tile = document.createElement('div');
  tile.className = 'photo-tile';
  tile.setAttribute('data-testid', 'photo-tile');
  tile.setAttribute('data-photo-id', photo.photo_id || photo.id);

  const thumbnailContainer = document.createElement('div');
  thumbnailContainer.className = 'photo-thumbnail-container';

  const thumbnail = document.createElement('img');
  thumbnail.className = 'photo-thumbnail';
  thumbnail.setAttribute('data-testid', 'photo-thumbnail');
  thumbnail.loading = 'lazy';

  if (photo.thumbnail_blob) {
    thumbnail.src = photo.thumbnail_blob;
  } else {
    thumbnail.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect width="200" height="200" fill="%23ddd"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%23999"%3ENo Image%3C/text%3E%3C/svg%3E';
  }

  thumbnail.alt = photo.file_path || 'Photo';

  thumbnail.addEventListener('error', () => {
    thumbnail.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect width="200" height="200" fill="%23ddd"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" fill="%23999"%3EError%3C/text%3E%3C/svg%3E';
  });

  thumbnailContainer.appendChild(thumbnail);

  const overlay = document.createElement('div');
  overlay.className = 'photo-overlay';
  overlay.setAttribute('data-testid', 'photo-metadata');
  overlay.style.display = 'none';

  const metadata = document.createElement('div');
  metadata.className = 'photo-metadata-content';

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

  if (photo.file_size) {
    const size = document.createElement('div');
    size.className = 'photo-size';
    size.textContent = formatFileSize(photo.file_size);
    metadata.appendChild(size);
  }

  overlay.appendChild(metadata);
  tile.appendChild(thumbnailContainer);
  tile.appendChild(overlay);

  tile.addEventListener('mouseenter', () => {
    overlay.style.display = 'flex';
  });

  tile.addEventListener('mouseleave', () => {
    overlay.style.display = 'none';
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

function formatFileSize(bytes) {
  if (!bytes || bytes === 0) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB'];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${units[i]}`;
}
