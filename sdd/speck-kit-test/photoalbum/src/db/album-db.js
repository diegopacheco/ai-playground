import { executeQuery, executeUpdate, executeInsert } from './database.js';

export function createAlbum(name) {
  if (!name || name.trim().length === 0) {
    throw new Error('Album name is required');
  }

  const maxOrderResult = executeQuery('SELECT COALESCE(MAX(display_order), -1) as max_order FROM Album');
  const nextOrder = maxOrderResult.values[0][0] + 1;

  const sql = `
    INSERT INTO Album (name, created_date, display_order)
    VALUES (?, datetime('now'), ?)
  `;

  return executeInsert(sql, [name.trim(), nextOrder]);
}

export function getAlbumById(id) {
  const sql = 'SELECT * FROM Album WHERE id = ?';
  const result = executeQuery(sql, [id]);

  if (result.values.length === 0) {
    return null;
  }

  const row = result.values[0];
  return {
    id: row[0],
    name: row[1],
    created_date: row[2],
    album_date: row[3],
    display_order: row[4],
    date_group: row[5],
    thumbnail_ref: row[6]
  };
}

export function getAllAlbums() {
  const sql = 'SELECT * FROM Album ORDER BY display_order ASC';
  const result = executeQuery(sql);

  return result.values.map(row => ({
    id: row[0],
    name: row[1],
    created_date: row[2],
    album_date: row[3],
    display_order: row[4],
    date_group: row[5],
    thumbnail_ref: row[6]
  }));
}

export function updateAlbumDisplayOrder(id, newOrder) {
  const sql = 'UPDATE Album SET display_order = ? WHERE id = ?';
  return executeUpdate(sql, [newOrder, id]);
}

export function deleteAlbum(id) {
  const sql = 'DELETE FROM Album WHERE id = ?';
  return executeUpdate(sql, [id]);
}

export function getAlbumPhotoRelations(albumId) {
  const sql = `
    SELECT ap.*, p.file_path, p.thumbnail_blob, p.date_taken, p.width, p.height
    FROM AlbumPhoto ap
    JOIN Photo p ON ap.photo_id = p.id
    WHERE ap.album_id = ?
    ORDER BY ap.order_index ASC, ap.added_date DESC
  `;

  const result = executeQuery(sql, [albumId]);

  return result.values.map(row => ({
    album_id: row[0],
    photo_id: row[1],
    order_index: row[2],
    added_date: row[3],
    file_path: row[4],
    thumbnail_blob: row[5],
    date_taken: row[6],
    width: row[7],
    height: row[8]
  }));
}

export function addPhotoToAlbum(albumId, photoId) {
  const albumExists = getAlbumById(albumId);
  if (!albumExists) {
    throw new Error(`Album with id ${albumId} does not exist`);
  }

  const maxOrderResult = executeQuery(
    'SELECT COALESCE(MAX(order_index), -1) as max_order FROM AlbumPhoto WHERE album_id = ?',
    [albumId]
  );
  const nextOrder = maxOrderResult.values[0][0] + 1;

  const sql = `
    INSERT INTO AlbumPhoto (album_id, photo_id, order_index, added_date)
    VALUES (?, ?, ?, datetime('now'))
  `;

  return executeInsert(sql, [albumId, photoId, nextOrder]);
}

export function removePhotoFromAlbum(albumId, photoId) {
  const sql = 'DELETE FROM AlbumPhoto WHERE album_id = ? AND photo_id = ?';
  return executeUpdate(sql, [albumId, photoId]);
}
