import { createAlbumCard } from './album-card.js';
import { DragDropHandler } from '../utils/drag-drop-handler.js';
import { createDateGroupHeader } from './date-group-header.js';

export function createAlbumGrid(albums, onAlbumClick, onReorder = null, grouped = false, onDelete = null) {
  const grid = document.createElement('div');
  grid.className = 'album-grid';
  grid.setAttribute('data-testid', 'album-grid');

  if (grouped && albums.length > 0 && albums[0].albums) {
    albums.forEach(group => {
      const header = createDateGroupHeader(group.dateGroup);
      grid.appendChild(header);

      const dateGroupDiv = document.createElement('div');
      dateGroupDiv.className = 'date-group';
      dateGroupDiv.setAttribute('data-testid', 'date-group');

      renderAlbumCards(dateGroupDiv, group.albums, onAlbumClick, onReorder, onDelete);

      grid.appendChild(dateGroupDiv);
    });

    return grid;
  }

  if (albums.length === 0) {
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.setAttribute('data-testid', 'empty-state');
    emptyState.innerHTML = `
      <p>No albums yet. Create your first album to get started!</p>
    `;
    grid.appendChild(emptyState);
    return grid;
  }

  renderAlbumCards(grid, albums, onAlbumClick, onReorder, onDelete);

  return grid;
}

function renderAlbumCards(container, albums, onAlbumClick, onReorder, onDelete) {
  let dragDropHandler = null;

  if (onReorder) {
    dragDropHandler = new DragDropHandler(onReorder);
    dragDropHandler.setAlbums(albums);

    const dragHandlers = {
      onDragStart: (e, element, album) => {
        const index = albums.findIndex(a => a.id === album.id);
        dragDropHandler.handleDragStart(element, index, album.id);
      },
      onDragEnd: (e, element) => {
        dragDropHandler.handleDragEnd(element);
      },
      onDragOver: (e, element) => {
        const album = albums.find(a => a.id === parseInt(element.getAttribute('data-album-id')));
        const index = albums.findIndex(a => a.id === album.id);
        dragDropHandler.handleDragOver(e, element, index);
      },
      onDragEnter: (e, element) => {
        dragDropHandler.handleDragEnter(e, element);
      },
      onDragLeave: (e, element) => {
        dragDropHandler.handleDragLeave(e, element);
      },
      onDrop: async (e, element) => {
        const album = albums.find(a => a.id === parseInt(element.getAttribute('data-album-id')));
        const index = albums.findIndex(a => a.id === album.id);
        await dragDropHandler.handleDrop(e, index);
      }
    };

    albums.forEach(album => {
      const card = createAlbumCard(album, onAlbumClick, dragHandlers, onDelete);
      container.appendChild(card);
    });
  } else {
    albums.forEach(album => {
      const card = createAlbumCard(album, onAlbumClick, null, onDelete);
      container.appendChild(card);
    });
  }
}

