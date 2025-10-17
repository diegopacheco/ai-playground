export function showConfirmModal(message, onConfirm, onCancel) {
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.setAttribute('data-testid', 'confirm-modal');

  const modalContent = document.createElement('div');
  modalContent.className = 'modal-content';

  const messageElement = document.createElement('p');
  messageElement.className = 'confirm-message';
  messageElement.textContent = message;

  const buttonGroup = document.createElement('div');
  buttonGroup.className = 'button-group';

  const cancelButton = document.createElement('button');
  cancelButton.className = 'btn btn-secondary';
  cancelButton.textContent = 'Cancel';
  cancelButton.setAttribute('data-testid', 'confirm-cancel');

  const confirmButton = document.createElement('button');
  confirmButton.className = 'btn btn-danger';
  confirmButton.textContent = 'Delete';
  confirmButton.setAttribute('data-testid', 'confirm-delete');

  buttonGroup.appendChild(cancelButton);
  buttonGroup.appendChild(confirmButton);

  modalContent.appendChild(messageElement);
  modalContent.appendChild(buttonGroup);
  overlay.appendChild(modalContent);

  const closeModal = () => {
    overlay.remove();
  };

  cancelButton.addEventListener('click', () => {
    closeModal();
    if (onCancel) {
      onCancel();
    }
  });

  confirmButton.addEventListener('click', async () => {
    closeModal();
    if (onConfirm) {
      await onConfirm();
    }
  });

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      closeModal();
      if (onCancel) {
        onCancel();
      }
    }
  });

  document.body.appendChild(overlay);
}
