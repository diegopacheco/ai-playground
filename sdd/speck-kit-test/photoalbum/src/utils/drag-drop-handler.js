export class DragDropHandler {
  constructor(onReorder) {
    this.onReorder = onReorder;
    this.draggedElement = null;
    this.draggedIndex = null;
    this.albums = [];
  }

  setAlbums(albums) {
    this.albums = albums;
  }

  handleDragStart(element, index, albumId) {
    this.draggedElement = element;
    this.draggedIndex = index;

    element.classList.add('dragging');
    element.style.opacity = '0.5';

    const dragData = {
      albumId: albumId,
      sourceIndex: index
    };

    if (element.dataTransfer) {
      element.dataTransfer.effectAllowed = 'move';
      element.dataTransfer.setData('application/json', JSON.stringify(dragData));
    }
  }

  handleDragEnd(element) {
    element.classList.remove('dragging');
    element.style.opacity = '1';

    const dropIndicators = document.querySelectorAll('.drop-indicator');
    dropIndicators.forEach(indicator => indicator.remove());

    this.draggedElement = null;
    this.draggedIndex = null;
  }

  handleDragOver(event, targetElement, targetIndex) {
    event.preventDefault();

    if (this.draggedElement === targetElement) {
      return;
    }

    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'move';
    }

    this.showDropIndicator(targetElement, targetIndex);
  }

  handleDragEnter(event, targetElement) {
    event.preventDefault();

    if (this.draggedElement !== targetElement) {
      targetElement.classList.add('drag-over');
    }
  }

  handleDragLeave(event, targetElement) {
    targetElement.classList.remove('drag-over');
  }

  async handleDrop(event, targetIndex) {
    event.preventDefault();
    event.stopPropagation();

    const targetElement = event.currentTarget;
    targetElement.classList.remove('drag-over');

    if (this.draggedIndex === null || this.draggedIndex === targetIndex) {
      return;
    }

    const newOrders = this.calculateNewOrders(this.draggedIndex, targetIndex);

    if (this.onReorder) {
      await this.onReorder(newOrders);
    }
  }

  calculateNewOrders(sourceIndex, targetIndex) {
    const reordered = [...this.albums];
    const [movedItem] = reordered.splice(sourceIndex, 1);
    reordered.splice(targetIndex, 0, movedItem);

    return reordered.map((album, index) => ({
      albumId: album.id,
      displayOrder: index
    }));
  }

  showDropIndicator(targetElement, targetIndex) {
    const existingIndicator = document.querySelector('.drop-indicator');
    if (existingIndicator) {
      existingIndicator.remove();
    }

    if (this.draggedIndex === targetIndex) {
      return;
    }

    const indicator = document.createElement('div');
    indicator.className = 'drop-indicator';

    const rect = targetElement.getBoundingClientRect();
    const isAfter = targetIndex > this.draggedIndex;

    indicator.style.position = 'absolute';
    indicator.style.left = `${rect.left}px`;
    indicator.style.width = `${rect.width}px`;
    indicator.style.height = '4px';
    indicator.style.backgroundColor = 'var(--primary-color)';
    indicator.style.borderRadius = '2px';
    indicator.style.zIndex = '1000';
    indicator.style.pointerEvents = 'none';

    if (isAfter) {
      indicator.style.top = `${rect.bottom}px`;
    } else {
      indicator.style.top = `${rect.top - 4}px`;
    }

    document.body.appendChild(indicator);
  }
}

export function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
