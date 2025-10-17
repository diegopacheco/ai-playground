import exifr from 'exifr';

export async function extractPhotoMetadata(file) {
  try {
    const exifData = await exifr.parse(file, {
      pick: ['DateTimeOriginal', 'DateTime', 'CreateDate', 'DateCreated', 'Make', 'Model'],
      translateValues: false,
      reviveValues: false
    });

    const dateTaken = extractDate(exifData, file);
    const dimensions = await getImageDimensions(file);

    return {
      dateTaken,
      width: dimensions.width,
      height: dimensions.height,
      fileSize: file.size,
      mimeType: file.type,
      fileName: file.name,
      exifData
    };
  } catch (error) {
    console.warn('EXIF extraction failed, using fallback:', error);
    return getFallbackMetadata(file);
  }
}

function extractDate(exifData, file) {
  if (!exifData) {
    return new Date(file.lastModified).toISOString();
  }

  const dateFields = ['DateTimeOriginal', 'DateTime', 'CreateDate', 'DateCreated'];

  for (const field of dateFields) {
    if (exifData[field]) {
      try {
        const date = new Date(exifData[field]);
        if (!isNaN(date.getTime())) {
          return date.toISOString();
        }
      } catch (e) {
        continue;
      }
    }
  }

  return new Date(file.lastModified).toISOString();
}

async function getImageDimensions(file) {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve({
        width: img.naturalWidth,
        height: img.naturalHeight
      });
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve({ width: null, height: null });
    };

    img.src = url;
  });
}

function getFallbackMetadata(file) {
  return {
    dateTaken: new Date(file.lastModified).toISOString(),
    width: null,
    height: null,
    fileSize: file.size,
    mimeType: file.type,
    fileName: file.name,
    exifData: null
  };
}

export function formatDateForDisplay(isoDateString) {
  if (!isoDateString) return 'Unknown Date';

  try {
    const date = new Date(isoDateString);
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    }).format(date);
  } catch (e) {
    return 'Unknown Date';
  }
}
