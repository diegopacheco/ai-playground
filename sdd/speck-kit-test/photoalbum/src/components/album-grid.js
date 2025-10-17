import { createAlbumCard } from './album-card.js';
import { DragDropHandler } from '../utils/drag-drop-handler.js';

export function createAlbumGrid(albums, onAlbumClick, onReorder = null) {
  const grid = document.createElement('div');
  grid.className = 'album-grid';
  grid.setAttribute('data-testid', 'album-grid');

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
      const card = createAlbumCard(album, onAlbumClick, dragHandlers);
      grid.appendChild(card);
    });
  } else {
    albums.forEach(album => {
      const card = createAlbumCard(album, onAlbumClick);
      grid.appendChild(card);
    });
  }

  return grid;
}

