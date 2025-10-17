import { executeQuery, executeInsert, executeUpdate } from './database.js';

export function createPhoto(photoData) {
  const {
    filePath,
    fileHandleRef = null,
    dateTaken = null,
    width = null,
    height = null,
    fileSize = null,
    mimeType = 'image/jpeg',
    thumbnailBlob = null
  } = photoData;

  if (!filePath || filePath.trim().length === 0) {
    throw new Error('File path is required');
  }

  const sql = `
    INSERT INTO Photo (
      file_path,
      file_handle_ref,
      date_taken,
      width,
      height,
      file_size,
      mime_type,
      thumbnail_blob,
      created_date
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
  `;

  return executeInsert(sql, [
    filePath.trim(),
    fileHandleRef,
    dateTaken,
    width,
    height,
    fileSize,
    mimeType,
    thumbnailBlob
  ]);
}

export function getPhotoById(id) {
  const sql = 'SELECT * FROM Photo WHERE id = ?';
  const result = executeQuery(sql, [id]);

  if (result.values.length === 0) {
    return null;
  }

  const row = result.values[0];
  return {
    id: row[0],
    file_path: row[1],
    file_handle_ref: row[2],
    date_taken: row[3],
    width: row[4],
    height: row[5],
    file_size: row[6],
    mime_type: row[7],
    thumbnail_blob: row[8],
    created_date: row[9]
  };
}

export function getPhotoByPath(filePath) {
  const sql = 'SELECT * FROM Photo WHERE file_path = ?';
  const result = executeQuery(sql, [filePath]);

  if (result.values.length === 0) {
    return null;
  }

  const row = result.values[0];
  return {
    id: row[0],
    file_path: row[1],
    file_handle_ref: row[2],
    date_taken: row[3],
    width: row[4],
    height: row[5],
    file_size: row[6],
    mime_type: row[7],
    thumbnail_blob: row[8],
    created_date: row[9]
  };
}

export function getAllPhotos() {
  const sql = 'SELECT * FROM Photo ORDER BY date_taken DESC NULLS LAST, created_date DESC';
  const result = executeQuery(sql);

  return result.values.map(row => ({
    id: row[0],
    file_path: row[1],
    file_handle_ref: row[2],
    date_taken: row[3],
    width: row[4],
    height: row[5],
    file_size: row[6],
    mime_type: row[7],
    thumbnail_blob: row[8],
    created_date: row[9]
  }));
}

export function updatePhoto(id, updates) {
  const allowedFields = [
    'file_handle_ref',
    'date_taken',
    'width',
    'height',
    'file_size',
    'mime_type',
    'thumbnail_blob'
  ];

  const fields = [];
  const values = [];

  for (const [key, value] of Object.entries(updates)) {
    if (allowedFields.includes(key)) {
      fields.push(`${key} = ?`);
      values.push(value);
    }
  }

  if (fields.length === 0) {
    throw new Error('No valid fields to update');
  }

  values.push(id);

  const sql = `UPDATE Photo SET ${fields.join(', ')} WHERE id = ?`;
  return executeUpdate(sql, values);
}

export function deletePhoto(id) {
  const sql = 'DELETE FROM Photo WHERE id = ?';
  return executeUpdate(sql, [id]);
}

export function getPhotosNotInAnyAlbum() {
  const sql = `
    SELECT p.*
    FROM Photo p
    LEFT JOIN AlbumPhoto ap ON p.id = ap.photo_id
    WHERE ap.photo_id IS NULL
    ORDER BY p.date_taken DESC NULLS LAST, p.created_date DESC
  `;

  const result = executeQuery(sql);

  return result.values.map(row => ({
    id: row[0],
    file_path: row[1],
    file_handle_ref: row[2],
    date_taken: row[3],
    width: row[4],
    height: row[5],
    file_size: row[6],
    mime_type: row[7],
    thumbnail_blob: row[8],
    created_date: row[9]
  }));
}
