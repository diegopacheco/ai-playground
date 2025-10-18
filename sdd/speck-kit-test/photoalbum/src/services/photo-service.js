import {
  createPhoto as dbCreatePhoto,
  getPhotoById as dbGetPhotoById,
  getPhotoByPath,
  getAllPhotos as dbGetAllPhotos,
  updatePhoto as dbUpdatePhoto,
  deletePhoto as dbDeletePhoto,
  getPhotosNotInAnyAlbum as dbGetPhotosNotInAnyAlbum
} from '../db/photo-db.js';
import { extractPhotoMetadata } from '../utils/exif-reader.js';
import { generateThumbnail, blobToBase64 } from '../utils/thumbnail-generator.js';

export async function addPhoto(photoData) {
  const { filePath, thumbnail, dateTaken, width, height, fileSize, mimeType } = photoData;

  if (!filePath || filePath.trim().length === 0) {
    throw new Error('File path is required');
  }

  if (!thumbnail) {
    throw new Error('Thumbnail is required');
  }

  const existing = getPhotoByPath(filePath);
  if (existing) {
    throw new Error(`Photo with path ${filePath} already exists`);
  }

  return dbCreatePhoto({
    filePath,
    dateTaken,
    width,
    height,
    fileSize,
    mimeType: mimeType || 'image/jpeg',
    thumbnailBlob: thumbnail
  });
}

export async function processAndAddPhoto(file) {
  const metadata = await extractPhotoMetadata(file);

  const thumbnailData = await generateThumbnail(file);
  const thumbnailBase64 = await blobToBase64(thumbnailData.blob);

  return addPhoto({
    filePath: file.name,
    thumbnail: thumbnailBase64,
    dateTaken: metadata.dateTaken,
    width: metadata.width,
    height: metadata.height,
    fileSize: metadata.fileSize,
    mimeType: metadata.mimeType
  });
}

export async function getPhotoById(id) {
  const photo = dbGetPhotoById(id);

  if (!photo) {
    return null;
  }

  return photo;
}

export async function getAllPhotos() {
  return dbGetAllPhotos();
}

export async function getPhotosNotInAnyAlbum() {
  return dbGetPhotosNotInAnyAlbum();
}

export async function updatePhotoMetadata(id, updates) {
  const photo = dbGetPhotoById(id);
  if (!photo) {
    throw new Error(`Photo with id ${id} not found`);
  }

  return dbUpdatePhoto(id, updates);
}

export async function deletePhoto(id) {
  const photo = dbGetPhotoById(id);
  if (!photo) {
    throw new Error(`Photo with id ${id} not found`);
  }

  return dbDeletePhoto(id);
}

export async function batchAddPhotos(files) {
  const results = {
    success: [],
    failed: []
  };

  for (const fileWrapper of files) {
    try {
      const photoId = await processAndAddPhoto(fileWrapper.file);
      results.success.push({ file: fileWrapper.name, photoId });
    } catch (error) {
      results.failed.push({ file: fileWrapper.name, error: error.message });
    }
  }

  return results;
}
