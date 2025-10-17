import { selectPhotos } from '../utils/file-handler.js';

export function createAddPhotosButton(onPhotosSelected) {
  const button = document.createElement('button');
  button.className = 'btn btn-primary';
  button.setAttribute('data-testid', 'add-photos-button');
  button.textContent = 'Add Photos';

  button.addEventListener('click', async () => {
    try {
      button.disabled = true;
      button.textContent = 'Selecting...';

      const files = await selectPhotos(true);

      if (files && files.length > 0) {
        button.textContent = 'Processing...';

        const loadingIndicator = showLoadingIndicator(files.length);

        try {
          await onPhotosSelected(files);
        } finally {
          if (loadingIndicator) {
            loadingIndicator.remove();
          }
        }
      }

      button.disabled = false;
      button.textContent = 'Add Photos';
    } catch (error) {
      console.error('Error selecting photos:', error);

      if (error.name !== 'AbortError') {
        showErrorMessage(error.message || 'Failed to add photos');
      }

      button.disabled = false;
      button.textContent = 'Add Photos';
    }
  });

  return button;
}

function showLoadingIndicator(photoCount) {
  const indicator = document.createElement('div');
  indicator.className = 'loading-indicator';
  indicator.setAttribute('data-testid', 'loading-indicator');

  const spinner = document.createElement('div');
  spinner.className = 'spinner';

  const message = document.createElement('div');
  message.className = 'loading-message';
  message.textContent = `Processing ${photoCount} photo${photoCount > 1 ? 's' : ''}...`;

  indicator.appendChild(spinner);
  indicator.appendChild(message);

  document.body.appendChild(indicator);

  return indicator;
}

function showErrorMessage(message) {
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error-message toast';
  errorDiv.setAttribute('data-testid', 'error-message');
  errorDiv.textContent = message;

  document.body.appendChild(errorDiv);

  setTimeout(() => {
    errorDiv.classList.add('show');
  }, 10);

  setTimeout(() => {
    errorDiv.classList.remove('show');
    setTimeout(() => errorDiv.remove(), 300);
  }, 3000);
}
