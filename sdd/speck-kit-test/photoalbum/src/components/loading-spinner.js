export function createLoadingSpinner(message = 'Loading...') {
  const container = document.createElement('div');
  container.className = 'loading-container';
  container.setAttribute('data-testid', 'loading-spinner');
  container.setAttribute('role', 'status');
  container.setAttribute('aria-live', 'polite');

  const spinner = document.createElement('div');
  spinner.className = 'spinner';

  const text = document.createElement('p');
  text.className = 'loading-text';
  text.textContent = message;

  container.appendChild(spinner);
  container.appendChild(text);

  return container;
}

export function showLoading(container, message = 'Loading...') {
  const loadingSpinner = createLoadingSpinner(message);
  container.innerHTML = '';
  container.appendChild(loadingSpinner);
}

export function hideLoading(container) {
  const loadingSpinner = container.querySelector('.loading-container');
  if (loadingSpinner) {
    loadingSpinner.remove();
  }
}
