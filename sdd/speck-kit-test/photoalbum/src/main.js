import { initApp } from './db/database.js';
import { createAlbum, getAllAlbums, getAlbumPhotos, addPhotoToAlbum, reorderAlbums, getAlbumsGroupedByDate, deleteAlbum } from './services/album-service.js';
import { batchAddPhotos } from './services/photo-service.js';
import { createAlbumGrid } from './components/album-grid.js';
import { showCreateAlbumModal } from './components/create-album-modal.js';
import { createPhotoTileGrid } from './components/photo-tile-grid.js';
import { createAddPhotosButton } from './components/add-photos-button.js';
import { createGroupingToggle, getGroupingPreference } from './components/grouping-toggle.js';
import { showLoading } from './components/loading-spinner.js';
import { showConfirmModal } from './components/confirm-modal.js';

let currentAlbum = null;
let currentView = 'albums';
let isGroupedView = false;

async function init() {
  try {
    console.log('Initializing Photo Album Organizer...');

    await initApp();

    console.log('Database initialized');

    isGroupedView = getGroupingPreference();

    await loadAlbumsView();

    setupEventListeners();

    console.log('Application ready');
  } catch (error) {
    console.error('Failed to initialize application:', error);
    showError('Failed to initialize application. Please refresh the page.');
  }
}

function setupEventListeners() {
  const createAlbumBtn = document.getElementById('create-album-btn');

  if (createAlbumBtn) {
    createAlbumBtn.addEventListener('click', handleCreateAlbum);
  }
}

async function handleCreateAlbum() {
  showCreateAlbumModal(
    async (albumName) => {
      try {
        await createAlbum(albumName);
        await loadAlbumsView();
      } catch (error) {
        console.error('Error creating album:', error);
        throw error;
      }
    },
    () => {
      console.log('Album creation cancelled');
    }
  );
}

async function loadAlbumsView() {
  currentView = 'albums';
  currentAlbum = null;

  const appContainer = document.getElementById('app');
  const header = document.querySelector('header h1');

  if (header) {
    header.textContent = 'Photo Album Organizer';
  }

  const backButton = document.getElementById('back-to-albums');
  if (backButton) {
    backButton.style.display = 'none';
  }

  const createButton = document.getElementById('create-album-btn');
  if (createButton) {
    createButton.style.display = 'inline-block';
  }

  let existingToggle = document.getElementById('grouping-toggle-btn');
  if (existingToggle) {
    existingToggle.remove();
  }

  const groupingToggle = createGroupingToggle(isGroupedView, handleGroupingToggle);
  groupingToggle.id = 'grouping-toggle-btn';

  const headerElement = document.querySelector('header');
  if (headerElement && createButton) {
    headerElement.insertBefore(groupingToggle, createButton);
  }

  try {
    if (appContainer) {
      showLoading(appContainer, 'Loading albums...');
    }

    let albumsData;

    if (isGroupedView) {
      albumsData = await getAlbumsGroupedByDate();
    } else {
      albumsData = await getAllAlbums();
    }

    const grid = createAlbumGrid(albumsData, handleAlbumClick, handleAlbumReorder, isGroupedView, handleDeleteAlbum);

    if (appContainer) {
      appContainer.innerHTML = '';
      appContainer.appendChild(grid);
    }
  } catch (error) {
    console.error('Error loading albums:', error);
    if (appContainer) {
      appContainer.innerHTML = '';
    }
    showError('Failed to load albums. Please try again.');
  }
}

async function handleGroupingToggle(newGroupedState) {
  isGroupedView = newGroupedState;
  await loadAlbumsView();
}

async function handleAlbumReorder(newOrders) {
  try {
    await reorderAlbums(newOrders);
    await loadAlbumsView();
  } catch (error) {
    console.error('Error reordering albums:', error);
    showError('Failed to reorder albums');
  }
}

async function handleDeleteAlbum(album) {
  showConfirmModal(
    `Are you sure you want to delete "${album.name}"? This will remove all photos from this album.`,
    async () => {
      try {
        await deleteAlbum(album.id);
        await loadAlbumsView();
        showSuccess(`Album "${album.name}" deleted successfully`);
      } catch (error) {
        console.error('Error deleting album:', error);
        showError('Failed to delete album. Please try again.');
      }
    }
  );
}

async function handleAlbumClick(album) {
  await loadAlbumView(album);
}

async function loadAlbumView(album) {
  currentView = 'album';
  currentAlbum = album;

  const appContainer = document.getElementById('app');
  const header = document.querySelector('header h1');

  if (header) {
    const albumHeader = document.createElement('div');
    albumHeader.setAttribute('data-testid', 'album-header');
    albumHeader.innerHTML = `
      <span>${album.name}</span>
    `;
    header.innerHTML = '';
    header.appendChild(albumHeader);
  }

  let backButton = document.getElementById('back-to-albums');
  if (!backButton) {
    backButton = document.createElement('button');
    backButton.id = 'back-to-albums';
    backButton.setAttribute('data-testid', 'back-to-albums');
    backButton.textContent = '← Back to Albums';
    backButton.className = 'btn btn-secondary';

    const headerElement = document.querySelector('header');
    if (headerElement) {
      headerElement.insertBefore(backButton, headerElement.firstChild);
    }
  }

  backButton.style.display = 'inline-block';
  backButton.onclick = loadAlbumsView;

  const createButton = document.getElementById('create-album-btn');
  if (createButton) {
    createButton.style.display = 'none';
  }

  try {
    if (appContainer) {
      showLoading(appContainer, 'Loading photos...');
    }

    const photos = await getAlbumPhotos(album.id);

    const albumView = document.createElement('div');
    albumView.className = 'album-view';

    const toolbar = document.createElement('div');
    toolbar.className = 'album-toolbar';

    const addPhotosBtn = createAddPhotosButton(async (files) => {
      await handleAddPhotos(album.id, files);
    });

    const photoCount = document.createElement('div');
    photoCount.className = 'photo-count';
    photoCount.setAttribute('data-testid', 'photo-count');
    photoCount.textContent = `${photos.length} photo${photos.length !== 1 ? 's' : ''}`;

    toolbar.appendChild(addPhotosBtn);
    toolbar.appendChild(photoCount);

    const photoGrid = createPhotoTileGrid(photos, handlePhotoClick);

    albumView.appendChild(toolbar);
    albumView.appendChild(photoGrid);

    if (appContainer) {
      appContainer.innerHTML = '';
      appContainer.appendChild(albumView);
    }
  } catch (error) {
    console.error('Error loading album:', error);
    if (appContainer) {
      appContainer.innerHTML = '';
    }
    showError('Failed to load album. Please try again.');
  }
}

async function handleAddPhotos(albumId, files) {
  try {
    const results = await batchAddPhotos(files);

    for (const success of results.success) {
      await addPhotoToAlbum(albumId, success.photoId);
    }

    if (results.failed.length > 0) {
      console.warn('Some photos failed to add:', results.failed);
      showError(`${results.failed.length} photo(s) failed to add`);
    }

    if (currentAlbum && currentAlbum.id === albumId) {
      await loadAlbumView(currentAlbum);
    }
  } catch (error) {
    console.error('Error adding photos:', error);
    showError('Failed to add photos');
  }
}

function handlePhotoClick(photo) {
  console.log('Photo clicked:', photo);
}

function showError(message) {
  showToast(message, 'error');
}

function showSuccess(message) {
  showToast(message, 'success');
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('show');
  }, 10);

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

document.addEventListener('DOMContentLoaded', init);
