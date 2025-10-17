const SUPPORTED_IMAGE_TYPES = [
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/gif',
  'image/webp',
  'image/heic',
  'image/heif'
];

export function isFileSystemAccessSupported() {
  return 'showOpenFilePicker' in window;
}

export async function selectPhotos(multiple = true) {
  if (!isFileSystemAccessSupported()) {
    throw new Error('File System Access API is not supported in this browser');
  }

  try {
    const fileHandles = await window.showOpenFilePicker({
      types: [
        {
          description: 'Images',
          accept: {
            'image/*': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.heif']
          }
        }
      ],
      multiple,
      excludeAcceptAllOption: false
    });

    const files = await Promise.all(
      fileHandles.map(async (handle) => {
        const file = await handle.getFile();
        return {
          file,
          handle,
          name: file.name,
          type: file.type,
          size: file.size
        };
      })
    );

    return files.filter(f => SUPPORTED_IMAGE_TYPES.includes(f.type));
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('User cancelled file selection');
      return [];
    }
    console.error('File selection failed:', error);
    throw new Error(`Failed to select photos: ${error.message}`);
  }
}

export async function requestFilePermission(fileHandle) {
  const permission = await fileHandle.queryPermission({ mode: 'read' });

  if (permission === 'granted') {
    return true;
  }

  if (permission === 'prompt') {
    const newPermission = await fileHandle.requestPermission({ mode: 'read' });
    return newPermission === 'granted';
  }

  return false;
}

export async function getFileFromHandle(fileHandle) {
  try {
    const hasPermission = await requestFilePermission(fileHandle);

    if (!hasPermission) {
      throw new Error('Permission denied to access file');
    }

    return await fileHandle.getFile();
  } catch (error) {
    console.error('Failed to get file from handle:', error);
    throw new Error(`Failed to access file: ${error.message}`);
  }
}

export function serializeFileHandle(fileHandle) {
  return JSON.stringify({
    kind: fileHandle.kind,
    name: fileHandle.name
  });
}

export async function verifyFileHandleAccess(fileHandle) {
  try {
    await fileHandle.getFile();
    return true;
  } catch (error) {
    console.warn('File handle no longer accessible:', error);
    return false;
  }
}

export function validateImageFile(file) {
  if (!SUPPORTED_IMAGE_TYPES.includes(file.type)) {
    throw new Error(`Unsupported file type: ${file.type}`);
  }

  const maxSize = 50 * 1024 * 1024;
  if (file.size > maxSize) {
    throw new Error(`File too large: ${(file.size / 1024 / 1024).toFixed(2)}MB (max 50MB)`);
  }

  return true;
}

export async function selectDirectory() {
  if (!('showDirectoryPicker' in window)) {
    throw new Error('Directory selection is not supported in this browser');
  }

  try {
    const directoryHandle = await window.showDirectoryPicker();
    return directoryHandle;
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('User cancelled directory selection');
      return null;
    }
    throw new Error(`Failed to select directory: ${error.message}`);
  }
}
