import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { initApp, executeQuery, closeDatabase, clearAllData } from '../../src/db/database.js';

describe('Photo Model Schema Contract', () => {
  beforeEach(async () => {
    await initApp();
  });

  afterEach(async () => {
    await clearAllData();
  });

  it('should have Photo table with correct columns', () => {
    const result = executeQuery('PRAGMA table_info(Photo)');
    const columns = result.values.map(col => ({
      name: col[1],
      type: col[2],
      notnull: col[3],
      pk: col[5]
    }));

    expect(columns).toContainEqual(expect.objectContaining({ name: 'id', type: 'INTEGER', pk: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'file_path', type: 'TEXT', notnull: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'file_handle_ref', type: 'TEXT' }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'date_taken', type: 'TEXT' }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'width', type: 'INTEGER' }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'height', type: 'INTEGER' }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'file_size', type: 'INTEGER' }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'mime_type', type: 'TEXT', notnull: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'thumbnail_blob', type: 'BLOB' }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'created_date', type: 'TEXT', notnull: 1 }));
  });

  it('should enforce NOT NULL constraint on file_path', () => {
    expect(() => {
      executeQuery('INSERT INTO Photo (file_path, created_date) VALUES (NULL, datetime("now"))');
    }).toThrow();
  });

  it('should enforce NOT NULL constraint on mime_type', () => {
    expect(() => {
      executeQuery('INSERT INTO Photo (file_path, mime_type, created_date) VALUES ("/path/to/photo.jpg", NULL, datetime("now"))');
    }).toThrow();
  });

  it('should have unique constraint on file_path', () => {
    executeQuery('INSERT INTO Photo (file_path, created_date) VALUES ("/path/photo.jpg", datetime("now"))');

    expect(() => {
      executeQuery('INSERT INTO Photo (file_path, created_date) VALUES ("/path/photo.jpg", datetime("now"))');
    }).toThrow();
  });

  it('should auto-increment photo id', () => {
    executeQuery('INSERT INTO Photo (file_path, created_date) VALUES ("/path/photo1.jpg", datetime("now"))');
    executeQuery('INSERT INTO Photo (file_path, created_date) VALUES ("/path/photo2.jpg", datetime("now"))');

    const photos = executeQuery('SELECT id FROM Photo ORDER BY id');
    expect(photos.values.length).toBe(2);
    expect(photos.values[1][0]).toBeGreaterThan(photos.values[0][0]);
  });

  it('should have idx_photo_date_taken index for date_taken', () => {
    const indexes = executeQuery('PRAGMA index_list(Photo)');
    const indexNames = indexes.values.map(idx => idx[1]);
    expect(indexNames).toContain('idx_photo_date_taken');
  });
});
