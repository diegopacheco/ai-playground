export function createAlbumModal(onSubmit, onCancel) {
  const modal = document.createElement('div');
  modal.className = 'modal-overlay';
  modal.setAttribute('data-testid', 'create-album-modal');

  const modalContent = document.createElement('div');
  modalContent.className = 'modal-content';

  const header = document.createElement('h2');
  header.textContent = 'Create New Album';

  const form = document.createElement('form');
  form.className = 'create-album-form';

  const inputGroup = document.createElement('div');
  inputGroup.className = 'input-group';

  const label = document.createElement('label');
  label.htmlFor = 'album-name';
  label.textContent = 'Album Name';

  const input = document.createElement('input');
  input.type = 'text';
  input.id = 'album-name';
  input.className = 'album-name-input';
  input.setAttribute('data-testid', 'album-name-input');
  input.placeholder = 'Enter album name';
  input.required = true;
  input.autofocus = true;

  const errorMessage = document.createElement('div');
  errorMessage.className = 'error-message';
  errorMessage.setAttribute('data-testid', 'album-name-error');
  errorMessage.style.display = 'none';

  inputGroup.appendChild(label);
  inputGroup.appendChild(input);
  inputGroup.appendChild(errorMessage);

  const buttonGroup = document.createElement('div');
  buttonGroup.className = 'button-group';

  const cancelButton = document.createElement('button');
  cancelButton.type = 'button';
  cancelButton.className = 'btn btn-secondary';
  cancelButton.setAttribute('data-testid', 'create-album-cancel');
  cancelButton.textContent = 'Cancel';

  const submitButton = document.createElement('button');
  submitButton.type = 'submit';
  submitButton.className = 'btn btn-primary';
  submitButton.setAttribute('data-testid', 'create-album-submit');
  submitButton.textContent = 'Create Album';

  buttonGroup.appendChild(cancelButton);
  buttonGroup.appendChild(submitButton);

  form.appendChild(inputGroup);
  form.appendChild(buttonGroup);

  modalContent.appendChild(header);
  modalContent.appendChild(form);

  modal.appendChild(modalContent);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const albumName = input.value.trim();

    if (!albumName) {
      errorMessage.textContent = 'Album name is required';
      errorMessage.style.display = 'block';
      input.focus();
      return;
    }

    try {
      submitButton.disabled = true;
      submitButton.textContent = 'Creating...';

      await onSubmit(albumName);

      closeModal();
    } catch (error) {
      errorMessage.textContent = error.message || 'Failed to create album';
      errorMessage.style.display = 'block';
      submitButton.disabled = false;
      submitButton.textContent = 'Create Album';
    }
  });

  cancelButton.addEventListener('click', () => {
    if (onCancel) onCancel();
    closeModal();
  });

  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      if (onCancel) onCancel();
      closeModal();
    }
  });

  function closeModal() {
    modal.remove();
  }

  input.addEventListener('input', () => {
    errorMessage.style.display = 'none';
  });

  return modal;
}

export function showCreateAlbumModal(onSubmit, onCancel) {
  const modal = createAlbumModal(onSubmit, onCancel);
  document.body.appendChild(modal);

  const input = modal.querySelector('input');
  if (input) {
    setTimeout(() => input.focus(), 100);
  }

  return modal;
}
