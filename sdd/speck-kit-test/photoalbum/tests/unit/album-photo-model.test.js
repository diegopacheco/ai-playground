import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { initApp, executeQuery, closeDatabase, clearAllData } from '../../src/db/database.js';

describe('AlbumPhoto Junction Table Schema Contract', () => {
  beforeEach(async () => {
    await initApp();
  });

  afterEach(async () => {
    await clearAllData();
  });

  it('should have AlbumPhoto table with correct columns', () => {
    const result = executeQuery('PRAGMA table_info(AlbumPhoto)');
    const columns = result.values.map(col => ({
      name: col[1],
      type: col[2],
      notnull: col[3],
      pk: col[5]
    }));

    expect(columns).toContainEqual(expect.objectContaining({ name: 'album_id', type: 'INTEGER', notnull: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'photo_id', type: 'INTEGER', notnull: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'order_index', type: 'INTEGER', notnull: 1 }));
    expect(columns).toContainEqual(expect.objectContaining({ name: 'added_date', type: 'TEXT', notnull: 1 }));
  });

  it('should have composite primary key on album_id and photo_id', () => {
    executeQuery('INSERT INTO Album (name, created_date, display_order) VALUES ("Test Album", datetime("now"), 1)');
    const albumResult = executeQuery('SELECT last_insert_rowid() as id');
    const albumId = albumResult.values[0][0];

    executeQuery('INSERT INTO Photo (file_path, created_date) VALUES ("/path/photo.jpg", datetime("now"))');
    const photoResult = executeQuery('SELECT last_insert_rowid() as id');
    const photoId = photoResult.values[0][0];

    executeQuery(`INSERT INTO AlbumPhoto (album_id, photo_id, added_date) VALUES (${albumId}, ${photoId}, datetime("now"))`);

    expect(() => {
      executeQuery(`INSERT INTO AlbumPhoto (album_id, photo_id, added_date) VALUES (${albumId}, ${photoId}, datetime("now"))`);
    }).toThrow();
  });

  it('should enforce foreign key constraint on album_id', () => {
    executeQuery('INSERT INTO Photo (file_path, created_date) VALUES ("/path/photo.jpg", datetime("now"))');
    const photoResult = executeQuery('SELECT last_insert_rowid() as id');
    const photoId = photoResult.values[0][0];

    expect(() => {
      executeQuery(`INSERT INTO AlbumPhoto (album_id, photo_id, added_date) VALUES (99999, ${photoId}, datetime("now"))`);
    }).toThrow();
  });

  it('should enforce foreign key constraint on photo_id', () => {
    executeQuery('INSERT INTO Album (name, created_date, display_order) VALUES ("Test Album", datetime("now"), 1)');
    const albumResult = executeQuery('SELECT last_insert_rowid() as id');
    const albumId = albumResult.values[0][0];

    expect(() => {
      executeQuery(`INSERT INTO AlbumPhoto (album_id, photo_id, added_date) VALUES (${albumId}, 99999, datetime("now"))`);
    }).toThrow();
  });

  it('should cascade delete when album is deleted', () => {
    executeQuery('INSERT INTO Album (name, created_date, display_order) VALUES ("Test Album", datetime("now"), 1)');
    const albumResult = executeQuery('SELECT last_insert_rowid() as id');
    const albumId = albumResult.values[0][0];

    executeQuery('INSERT INTO Photo (file_path, created_date) VALUES ("/path/photo.jpg", datetime("now"))');
    const photoResult = executeQuery('SELECT last_insert_rowid() as id');
    const photoId = photoResult.values[0][0];

    executeQuery(`INSERT INTO AlbumPhoto (album_id, photo_id, added_date) VALUES (${albumId}, ${photoId}, datetime("now"))`);

    executeQuery(`DELETE FROM Album WHERE id = ${albumId}`);

    const relations = executeQuery(`SELECT * FROM AlbumPhoto WHERE album_id = ${albumId}`);
    expect(relations.values.length).toBe(0);
  });

  it('should cascade delete when photo is deleted', () => {
    executeQuery('INSERT INTO Album (name, created_date, display_order) VALUES ("Test Album", datetime("now"), 1)');
    const albumResult = executeQuery('SELECT last_insert_rowid() as id');
    const albumId = albumResult.values[0][0];

    executeQuery('INSERT INTO Photo (file_path, created_date) VALUES ("/path/photo.jpg", datetime("now"))');
    const photoResult = executeQuery('SELECT last_insert_rowid() as id');
    const photoId = photoResult.values[0][0];

    executeQuery(`INSERT INTO AlbumPhoto (album_id, photo_id, added_date) VALUES (${albumId}, ${photoId}, datetime("now"))`);

    executeQuery(`DELETE FROM Photo WHERE id = ${photoId}`);

    const relations = executeQuery(`SELECT * FROM AlbumPhoto WHERE photo_id = ${photoId}`);
    expect(relations.values.length).toBe(0);
  });

  it('should have idx_album_photo_album index for album_id', () => {
    const indexes = executeQuery('PRAGMA index_list(AlbumPhoto)');
    const indexNames = indexes.values.map(idx => idx[1]);
    expect(indexNames).toContain('idx_album_photo_album');
  });

  it('should have idx_album_photo_photo index for photo_id', () => {
    const indexes = executeQuery('PRAGMA index_list(AlbumPhoto)');
    const indexNames = indexes.values.map(idx => idx[1]);
    expect(indexNames).toContain('idx_album_photo_photo');
  });
});
