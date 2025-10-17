import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { initApp, executeQuery, closeDatabase, clearAllData } from '../../src/db/database.js';

describe('Album Model Schema Contract', () => {
  beforeEach(async () => {
    await initApp();
  });

  afterEach(async () => {
    await clearAllData();
  });

  it('should have Album table with correct columns', () => {
    const result = executeQuery('PRAGMA table_info(Album)');
    const columns = result.values.map(col => ({
      name: col[1],
      type: col[2],
      notnull: col[3],
      pk: col[5]
    }));

    expect(columns).toContainEqual(expect.objectContaining({ name: 'id', type: 'INTEGER', pk: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'name', type: 'TEXT', notnull: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'created_date', type: 'TEXT', notnull: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'album_date', type: 'TEXT' }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'display_order', type: 'INTEGER', notnull: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'date_group', type: 'TEXT' }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'thumbnail_ref', type: 'INTEGER' }));
  });

  it('should enforce NOT NULL constraint on album name', () => {
    expect(() => {
      executeQuery('INSERT INTO Album (name, created_date, display_order) VALUES (NULL, datetime("now"), 1)');
    }).toThrow();
  });

  it('should enforce CHECK constraint on album name length', () => {
    expect(() => {
      executeQuery('INSERT INTO Album (name, created_date, display_order) VALUES ("", datetime("now"), 1)');
    }).toThrow();
  });

  it('should auto-increment album id', () => {
    const result1 = executeQuery('INSERT INTO Album (name, created_date, display_order) VALUES ("Album 1", datetime("now"), 1); SELECT last_insert_rowid() as id');
    const result2 = executeQuery('INSERT INTO Album (name, created_date, display_order) VALUES ("Album 2", datetime("now"), 2); SELECT last_insert_rowid() as id');

    const albums = executeQuery('SELECT id FROM Album ORDER BY id');
    expect(albums.values.length).toBe(2);
    expect(albums.values[1][0]).toBeGreaterThan(albums.values[0][0]);
  });

  it('should have idx_album_display_order index for display_order', () => {
    const indexes = executeQuery('PRAGMA index_list(Album)');
    const indexNames = indexes.values.map(idx => idx[1]);
    expect(indexNames).toContain('idx_album_display_order');
  });

  it('should have idx_album_date_group index for date_group', () => {
    const indexes = executeQuery('PRAGMA index_list(Album)');
    const indexNames = indexes.values.map(idx => idx[1]);
    expect(indexNames).toContain('idx_album_date_group');
  });
});
