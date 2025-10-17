# Data Model: Photo Album Organizer

**Created**: 2025-10-16
**Phase**: 1 (Design & Contracts)

## Overview

This document defines the data entities, relationships, and constraints for the photo album organizer application.

## Entity Relationship Diagram

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│     Album       │         │   AlbumPhoto     │         │      Photo      │
├─────────────────┤         ├──────────────────┤         ├─────────────────┤
│ id (PK)         │◄───────┤ album_id (FK)    │┌───────►│ id (PK)         │
│ name            │         │ photo_id (FK)    ││        │ file_path       │
│ created_date    │         │ order_index      ││        │ file_handle_ref │
│ album_date      │         │ added_date       ││        │ date_taken      │
│ display_order   │         └──────────────────┘│        │ width           │
│ date_group      │                             │        │ height          │
│ thumbnail_ref   │                             │        │ file_size       │
└─────────────────┘                             │        │ mime_type       │
                                                 │        │ thumbnail_blob  │
                                                 │        │ created_date    │
                                                 │        └─────────────────┘
                                                 │
                                                 └── Many-to-Many Relationship
```

## Entities

### 1. Album

Represents a photo album created by the user.

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique album identifier |
| name | TEXT | NOT NULL | User-defined album name |
| created_date | TEXT | NOT NULL, DEFAULT CURRENT_TIMESTAMP | ISO 8601 timestamp when album was created |
| album_date | TEXT | NULLABLE | Representative date for the album (earliest photo date) |
| display_order | INTEGER | NOT NULL, DEFAULT 0 | User-defined sort order for albums |
| date_group | TEXT | NULLABLE | Computed: YYYY-MM format for grouping (e.g., "2025-10") |
| thumbnail_ref | INTEGER | NULLABLE, FK(Photo.id) | Reference to photo used as album thumbnail |

**Indexes**:
- `idx_album_display_order` on (display_order)
- `idx_album_date_group` on (date_group)

**Validation Rules**:
- name MUST NOT be empty (min length: 1 character)
- display_order MUST be >= 0
- album_date format MUST be ISO 8601 (YYYY-MM-DDTHH:MM:SS.sssZ) if present
- date_group format MUST be YYYY-MM if computed

**Business Logic**:
- album_date is recalculated when photos are added/removed (earliest photo date)
- date_group is derived from album_date: strftime('%Y-%m', album_date)
- thumbnail_ref defaults to first photo in album if not explicitly set

### 2. Photo

Represents an individual photo file referenced by the application.

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique photo identifier |
| file_path | TEXT | NOT NULL, UNIQUE | Original file path on user's system |
| file_handle_ref | TEXT | NULLABLE | Serialized File System Access API handle |
| date_taken | TEXT | NULLABLE | Photo date from EXIF (DateTimeOriginal) |
| width | INTEGER | NULLABLE | Image width in pixels |
| height | INTEGER | NULLABLE | Image height in pixels |
| file_size | INTEGER | NULLABLE | File size in bytes |
| mime_type | TEXT | NOT NULL, DEFAULT 'image/jpeg' | MIME type (image/jpeg, image/png, etc.) |
| thumbnail_blob | BLOB | NULLABLE | Cached thumbnail data (600x600px JPEG) |
| created_date | TEXT | NOT NULL, DEFAULT CURRENT_TIMESTAMP | ISO 8601 timestamp when photo was added |

**Indexes**:
- `idx_photo_file_path` on (file_path) - UNIQUE
- `idx_photo_date_taken` on (date_taken)

**Validation Rules**:
- file_path MUST NOT be empty
- file_path MUST be unique across all photos
- date_taken format MUST be ISO 8601 if present
- mime_type MUST match pattern 'image/*'
- width, height, file_size MUST be > 0 if present

**Business Logic**:
- file_handle_ref is serialized FileSystemFileHandle (browser API)
- thumbnail_blob is generated lazily and cached
- date_taken extracted via exifr library from EXIF metadata
- Falls back to file.lastModified if no EXIF date

### 3. AlbumPhoto (Junction Table)

Represents the many-to-many relationship between albums and photos.

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| album_id | INTEGER | NOT NULL, FK(Album.id) ON DELETE CASCADE | Reference to album |
| photo_id | INTEGER | NOT NULL, FK(Photo.id) ON DELETE CASCADE | Reference to photo |
| order_index | INTEGER | NOT NULL, DEFAULT 0 | Sort order within the album |
| added_date | TEXT | NOT NULL, DEFAULT CURRENT_TIMESTAMP | When photo was added to this album |

**Primary Key**: Composite (album_id, photo_id)

**Indexes**:
- `idx_album_photo_album` on (album_id, order_index) - For album photo listing
- `idx_album_photo_photo` on (photo_id) - For reverse lookup

**Validation Rules**:
- (album_id, photo_id) combination MUST be unique
- order_index MUST be >= 0
- added_date format MUST be ISO 8601

**Business Logic**:
- order_index determines photo display order within album
- Same photo can exist in multiple albums (different album_id)
- Deleting album removes all AlbumPhoto entries (CASCADE)
- Deleting photo removes all AlbumPhoto entries (CASCADE)

## Relationships

### Album ↔ Photo (Many-to-Many)

- **Cardinality**: One album contains zero or more photos; one photo can belong to zero or more albums
- **Implementation**: Through AlbumPhoto junction table
- **Cascade Rules**:
  - Deleting an album: Remove all AlbumPhoto entries for that album (photos remain)
  - Deleting a photo: Remove all AlbumPhoto entries for that photo (albums remain)

### Album → Photo (Thumbnail Reference)

- **Cardinality**: One album has zero or one thumbnail photo
- **Implementation**: Album.thumbnail_ref foreign key to Photo.id
- **Cascade Rules**:
  - If thumbnail photo is deleted: Set Album.thumbnail_ref to NULL
  - Default behavior: Use first photo in album if thumbnail_ref is NULL

## State Transitions

### Album Lifecycle

```
[CREATE] → Empty Album (0 photos)
         ↓
      [ADD PHOTOS]
         ↓
      Active Album (1+ photos)
         ↓
      [REMOVE ALL PHOTOS]
         ↓
      Empty Album (0 photos)
         ↓
      [DELETE]
         ↓
      Deleted (removed from database)
```

### Photo Lifecycle

```
[SELECT FILE] → File Selected
         ↓
      [READ METADATA]
         ↓
      Photo Created (metadata extracted)
         ↓
      [ADD TO ALBUM(S)]
         ↓
      Photo In Album(s)
         ↓
      [REMOVE FROM ALL ALBUMS]
         ↓
      Orphaned Photo (0 albums)
         ↓
      [DELETE or AUTO-CLEANUP]
         ↓
      Deleted (removed from database)
```

## Computed Fields

### album_date (Album.album_date)

**Computation**:
```sql
SELECT MIN(p.date_taken)
FROM Photo p
JOIN AlbumPhoto ap ON ap.photo_id = p.id
WHERE ap.album_id = ?
```

**Trigger**: Recalculate when photos are added/removed from album

### date_group (Album.date_group)

**Computation**:
```sql
strftime('%Y-%m', album_date)
```

**Trigger**: Recalculate when album_date changes

### thumbnail_ref (Album.thumbnail_ref)

**Default Selection**:
```sql
SELECT ap.photo_id
FROM AlbumPhoto ap
WHERE ap.album_id = ?
ORDER BY ap.order_index ASC
LIMIT 1
```

**Trigger**: Use first photo if thumbnail_ref is NULL

## Data Integrity Constraints

1. **No Nested Albums**: Albums table has no self-referential foreign key
2. **Unique Photo Paths**: Photo.file_path is UNIQUE to prevent duplicates
3. **Orphan Prevention**: Photos not in any album can be auto-cleaned (optional)
4. **Referential Integrity**: All foreign keys use CASCADE for deletes
5. **Date Format**: All dates stored as ISO 8601 TEXT for SQLite compatibility
6. **Non-negative Ordering**: display_order and order_index >= 0

## Query Patterns

### 1. List All Albums with Photo Count (Ordered)
```sql
SELECT
  a.*,
  COUNT(ap.photo_id) as photo_count
FROM Album a
LEFT JOIN AlbumPhoto ap ON a.id = ap.album_id
GROUP BY a.id
ORDER BY a.display_order ASC
```

### 2. List Albums Grouped by Date
```sql
SELECT
  a.date_group,
  a.*,
  COUNT(ap.photo_id) as photo_count
FROM Album a
LEFT JOIN AlbumPhoto ap ON a.id = ap.album_id
WHERE a.date_group IS NOT NULL
GROUP BY a.id
ORDER BY a.date_group DESC, a.display_order ASC
```

### 3. Get Photos in Album (Ordered)
```sql
SELECT
  p.*,
  ap.order_index
FROM Photo p
JOIN AlbumPhoto ap ON p.id = ap.photo_id
WHERE ap.album_id = ?
ORDER BY ap.order_index ASC
```

### 4. Find Orphaned Photos (Not in Any Album)
```sql
SELECT p.*
FROM Photo p
LEFT JOIN AlbumPhoto ap ON p.id = ap.photo_id
WHERE ap.album_id IS NULL
```

## Performance Considerations

- **Indexes**: All foreign keys and sort columns are indexed
- **Lazy Loading**: Thumbnail blobs loaded only when needed (not in main queries)
- **Batch Inserts**: Use transactions for adding multiple photos
- **Virtual Scrolling**: Queries use LIMIT/OFFSET for large photo sets
- **Denormalization**: album_date and date_group are denormalized for query performance

## Migration Strategy

As this is the initial schema (v1), no migrations are needed. Future schema changes will be versioned and documented in a separate migrations/ directory.
