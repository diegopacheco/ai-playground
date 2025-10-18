const PLACEHOLDER_IMAGE = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"%3E%3Crect fill="%23f1f5f9" width="200" height="200"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dominant-baseline="middle" font-size="48" fill="%2394a3b8"%3E📷%3C/text%3E%3C/svg%3E';

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

  const observer = createIntersectionObserver();

  photos.forEach(photo => {
    const tile = createPhotoTile(photo, onPhotoClick, observer);
    grid.appendChild(tile);
  });

  return grid;
}

function createIntersectionObserver() {
  if (!('IntersectionObserver' in window)) {
    return null;
  }

  return new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        const src = img.getAttribute('data-src');
        if (src && !img.src.startsWith('data:image/') || img.src === PLACEHOLDER_IMAGE) {
          loadImageAsync(img, src);
        }
      }
    });
  }, {
    rootMargin: '50px',
    threshold: 0.01
  });
}

function loadImageAsync(img, src) {
  img.classList.remove('loaded');
  img.classList.add('loading');

  const tempImg = new Image();
  tempImg.onload = () => {
    img.src = src;
    img.classList.remove('loading');
    img.classList.add('loaded');
  };

  tempImg.onerror = () => {
    img.src = PLACEHOLDER_IMAGE;
    img.classList.remove('loading');
    img.classList.add('error', 'loaded');
  };

  tempImg.src = src;
}

function createPhotoTile(photo, onClick, observer) {
  const tile = document.createElement('div');
  tile.className = 'photo-tile';
  tile.setAttribute('data-testid', 'photo-tile');
  tile.setAttribute('data-photo-id', photo.photo_id || photo.id);

  const thumbnail = document.createElement('img');
  thumbnail.className = 'photo-thumbnail';
  thumbnail.setAttribute('data-testid', 'photo-thumbnail');
  thumbnail.alt = photo.file_path || 'Photo';
  thumbnail.decoding = 'async';

  const thumbnailSrc = photo.thumbnail_blob || PLACEHOLDER_IMAGE;

  if (observer && thumbnailSrc !== PLACEHOLDER_IMAGE) {
    thumbnail.src = PLACEHOLDER_IMAGE;
    thumbnail.classList.add('loaded');
    thumbnail.setAttribute('data-src', thumbnailSrc);
    observer.observe(thumbnail);
  } else {
    thumbnail.src = thumbnailSrc;
    if (thumbnailSrc === PLACEHOLDER_IMAGE) {
      thumbnail.classList.add('loaded');
    }
  }

  thumbnail.onerror = () => {
    thumbnail.src = PLACEHOLDER_IMAGE;
    thumbnail.classList.add('error');
  };

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
