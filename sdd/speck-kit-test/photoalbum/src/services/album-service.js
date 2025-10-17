import {
  createAlbum as dbCreateAlbum,
  getAlbumById,
  getAllAlbums as dbGetAllAlbums,
  getAlbumPhotoRelations,
  addPhotoToAlbum as dbAddPhotoToAlbum,
  removePhotoFromAlbum as dbRemovePhotoFromAlbum,
  updateAlbumDisplayOrder,
  deleteAlbum as dbDeleteAlbum
} from '../db/album-db.js';
import { getPhotoById } from '../db/photo-db.js';

export async function createAlbum(name) {
  if (!name || name.trim().length === 0) {
    throw new Error('Album name is required and cannot be empty');
  }

  return dbCreateAlbum(name);
}

export async function getAllAlbums() {
  return dbGetAllAlbums();
}

export async function getAlbum(id) {
  const album = getAlbumById(id);

  if (!album) {
    throw new Error(`Album with id ${id} not found`);
  }

  return album;
}

export async function addPhotoToAlbum(albumId, photoId) {
  const album = getAlbumById(albumId);
  if (!album) {
    throw new Error(`Album with id ${albumId} not found`);
  }

  const photo = getPhotoById(photoId);
  if (!photo) {
    throw new Error(`Photo with id ${photoId} not found`);
  }

  try {
    return dbAddPhotoToAlbum(albumId, photoId);
  } catch (error) {
    if (error.message.includes('UNIQUE constraint failed') || error.message.includes('PRIMARY KEY')) {
      throw new Error(`Photo ${photoId} is already in album ${albumId}`);
    }
    throw error;
  }
}

export async function getAlbumPhotos(albumId) {
  const album = getAlbumById(albumId);
  if (!album) {
    throw new Error(`Album with id ${albumId} not found`);
  }

  return getAlbumPhotoRelations(albumId);
}

export async function removePhotoFromAlbum(albumId, photoId) {
  return dbRemovePhotoFromAlbum(albumId, photoId);
}

export async function updateDisplayOrder(albumId, newOrder) {
  const album = getAlbumById(albumId);
  if (!album) {
    throw new Error(`Album with id ${albumId} not found`);
  }

  return updateAlbumDisplayOrder(albumId, newOrder);
}

export async function reorderAlbums(albumOrders) {
  for (const { albumId, displayOrder } of albumOrders) {
    await updateAlbumDisplayOrder(albumId, displayOrder);
  }
}

export async function deleteAlbum(id) {
  const album = getAlbumById(id);
  if (!album) {
    throw new Error(`Album with id ${id} not found`);
  }

  return dbDeleteAlbum(id);
}

export async function getAlbumsGroupedByDate() {
  const albums = dbGetAllAlbums();

  const grouped = {};

  for (const album of albums) {
    const group = album.date_group || 'Unknown Date';

    if (!grouped[group]) {
      grouped[group] = [];
    }

    grouped[group].push(album);
  }

  const sortedGroups = Object.keys(grouped).sort((a, b) => {
    if (a === 'Unknown Date') return 1;
    if (b === 'Unknown Date') return -1;
    return b.localeCompare(a);
  });

  return sortedGroups.map(group => ({
    dateGroup: group,
    albums: grouped[group]
  }));
}
