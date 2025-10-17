CREATE TABLE IF NOT EXISTS Album (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL CHECK(length(name) > 0),
  created_date TEXT NOT NULL DEFAULT (datetime('now')),
  album_date TEXT,
  display_order INTEGER NOT NULL DEFAULT 0 CHECK(display_order >= 0),
  date_group TEXT,
  thumbnail_ref INTEGER,
  FOREIGN KEY (thumbnail_ref) REFERENCES Photo(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS Photo (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_path TEXT NOT NULL UNIQUE CHECK(length(file_path) > 0),
  file_handle_ref TEXT,
  date_taken TEXT,
  width INTEGER CHECK(width IS NULL OR width > 0),
  height INTEGER CHECK(height IS NULL OR height > 0),
  file_size INTEGER CHECK(file_size IS NULL OR file_size > 0),
  mime_type TEXT NOT NULL DEFAULT 'image/jpeg' CHECK(mime_type LIKE 'image/%'),
  thumbnail_blob BLOB,
  created_date TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS AlbumPhoto (
  album_id INTEGER NOT NULL,
  photo_id INTEGER NOT NULL,
  order_index INTEGER NOT NULL DEFAULT 0 CHECK(order_index >= 0),
  added_date TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (album_id, photo_id),
  FOREIGN KEY (album_id) REFERENCES Album(id) ON DELETE CASCADE,
  FOREIGN KEY (photo_id) REFERENCES Photo(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_album_display_order ON Album(display_order);

CREATE INDEX IF NOT EXISTS idx_album_date_group ON Album(date_group);

CREATE INDEX IF NOT EXISTS idx_photo_file_path ON Photo(file_path);

CREATE INDEX IF NOT EXISTS idx_photo_date_taken ON Photo(date_taken);

CREATE INDEX IF NOT EXISTS idx_album_photo_album ON AlbumPhoto(album_id, order_index);

CREATE INDEX IF NOT EXISTS idx_album_photo_photo ON AlbumPhoto(photo_id);

CREATE TRIGGER IF NOT EXISTS update_album_date_on_photo_add
AFTER INSERT ON AlbumPhoto
BEGIN
  UPDATE Album
  SET album_date = (
    SELECT MIN(p.date_taken)
    FROM Photo p
    JOIN AlbumPhoto ap ON ap.photo_id = p.id
    WHERE ap.album_id = NEW.album_id AND p.date_taken IS NOT NULL
  ),
  date_group = strftime('%Y-%m', (
    SELECT MIN(p.date_taken)
    FROM Photo p
    JOIN AlbumPhoto ap ON ap.photo_id = p.id
    WHERE ap.album_id = NEW.album_id AND p.date_taken IS NOT NULL
  ))
  WHERE id = NEW.album_id;
END;

CREATE TRIGGER IF NOT EXISTS update_album_date_on_photo_remove
AFTER DELETE ON AlbumPhoto
BEGIN
  UPDATE Album
  SET album_date = (
    SELECT MIN(p.date_taken)
    FROM Photo p
    JOIN AlbumPhoto ap ON ap.photo_id = p.id
    WHERE ap.album_id = OLD.album_id AND p.date_taken IS NOT NULL
  ),
  date_group = strftime('%Y-%m', (
    SELECT MIN(p.date_taken)
    FROM Photo p
    JOIN AlbumPhoto ap ON ap.photo_id = p.id
    WHERE ap.album_id = OLD.album_id AND p.date_taken IS NOT NULL
  ))
  WHERE id = OLD.album_id;
END;

CREATE TRIGGER IF NOT EXISTS update_album_date_on_photo_date_change
AFTER UPDATE OF date_taken ON Photo
BEGIN
  UPDATE Album
  SET album_date = (
    SELECT MIN(p.date_taken)
    FROM Photo p
    JOIN AlbumPhoto ap ON ap.photo_id = p.id
    WHERE ap.album_id IN (
      SELECT album_id FROM AlbumPhoto WHERE photo_id = NEW.id
    ) AND p.date_taken IS NOT NULL
  ),
  date_group = strftime('%Y-%m', (
    SELECT MIN(p.date_taken)
    FROM Photo p
    JOIN AlbumPhoto ap ON ap.photo_id = p.id
    WHERE ap.album_id IN (
      SELECT album_id FROM AlbumPhoto WHERE photo_id = NEW.id
    ) AND p.date_taken IS NOT NULL
  ))
  WHERE id IN (
    SELECT album_id FROM AlbumPhoto WHERE photo_id = NEW.id
  );
END;
