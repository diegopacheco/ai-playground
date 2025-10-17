const THUMBNAIL_SIZE = 600;
const THUMBNAIL_QUALITY = 0.85;

export async function generateThumbnail(file) {
  try {
    const img = await loadImage(file);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const dimensions = calculateThumbnailDimensions(img.width, img.height, THUMBNAIL_SIZE);
    canvas.width = dimensions.width;
    canvas.height = dimensions.height;

    ctx.drawImage(img, 0, 0, dimensions.width, dimensions.height);

    const blob = await canvasToBlob(canvas, THUMBNAIL_QUALITY);

    return {
      blob,
      width: dimensions.width,
      height: dimensions.height,
      size: blob.size
    };
  } catch (error) {
    console.error('Thumbnail generation failed:', error);
    throw new Error(`Failed to generate thumbnail: ${error.message}`);
  }
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to load image'));
    };

    img.src = url;
  });
}

function calculateThumbnailDimensions(width, height, maxSize) {
  const aspectRatio = width / height;

  if (width > height) {
    return {
      width: Math.min(width, maxSize),
      height: Math.min(width, maxSize) / aspectRatio
    };
  } else {
    return {
      width: Math.min(height, maxSize) * aspectRatio,
      height: Math.min(height, maxSize)
    };
  }
}

function canvasToBlob(canvas, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to convert canvas to blob'));
        }
      },
      'image/jpeg',
      quality
    );
  });
}

export async function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export function base64ToBlob(base64String) {
  const parts = base64String.split(';base64,');
  const contentType = parts[0].split(':')[1];
  const raw = window.atob(parts[1]);
  const rawLength = raw.length;
  const uInt8Array = new Uint8Array(rawLength);

  for (let i = 0; i < rawLength; i++) {
    uInt8Array[i] = raw.charCodeAt(i);
  }

  return new Blob([uInt8Array], { type: contentType });
}

export function createThumbnailURL(blob) {
  return URL.createObjectURL(blob);
}

export function revokeThumbnailURL(url) {
  URL.revokeObjectURL(url);
}
