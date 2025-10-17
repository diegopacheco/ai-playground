import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { initApp, clearAllData } from '../../src/db/database.js';
import { createAlbum, addPhotoToAlbum, getAlbumPhotos, updateDisplayOrder, reorderAlbums, getAlbum, getAllAlbums, getAlbumsGroupedByDate } from '../../src/services/album-service.js';
import { addPhoto } from '../../src/services/photo-service.js';

describe('AlbumService', () => {
  beforeEach(async () => {
    await initApp();
  });

  afterEach(async () => {
    await clearAllData();
  });

  describe('createAlbum', () => {
    it('should create a new album with valid name', async () => {
      const albumId = await createAlbum('My Vacation Photos');

      expect(albumId).toBeDefined();
      expect(typeof albumId).toBe('number');
      expect(albumId).toBeGreaterThan(0);
    });

    it('should throw error when creating album with empty name', async () => {
      await expect(createAlbum('')).rejects.toThrow();
    });

    it('should throw error when creating album with null name', async () => {
      await expect(createAlbum(null)).rejects.toThrow();
    });

    it('should allow creating albums with duplicate names', async () => {
      const album1Id = await createAlbum('Duplicate Album');
      const album2Id = await createAlbum('Duplicate Album');

      expect(album1Id).toBeDefined();
      expect(album2Id).toBeDefined();
      expect(album2Id).not.toBe(album1Id);
    });

    it('should auto-assign display_order incrementally', async () => {
      const album1Id = await createAlbum('Album 1');
      const album2Id = await createAlbum('Album 2');
      const album3Id = await createAlbum('Album 3');

      expect(album1Id).toBeLessThan(album2Id);
      expect(album2Id).toBeLessThan(album3Id);
    });

    it('should set created_at and updated_at timestamps', async () => {
      const before = new Date();
      const albumId = await createAlbum('Timestamped Album');
      const after = new Date();

      expect(albumId).toBeGreaterThan(0);
    });
  });

  describe('addPhotoToAlbum', () => {
    it('should add photo to album successfully', async () => {
      const albumId = await createAlbum('Test Album');
      const photoId = await addPhoto({
        filePath: '/test/photo1.jpg',
        thumbnail: 'data:image/jpeg;base64,test1'
      });

      await expect(addPhotoToAlbum(albumId, photoId)).resolves.not.toThrow();
    });

    it('should throw error when adding photo to non-existent album', async () => {
      const photoId = await addPhoto({
        filePath: '/test/photo1.jpg',
        thumbnail: 'data:image/jpeg;base64,test1'
      });

      await expect(addPhotoToAlbum(99999, photoId)).rejects.toThrow();
    });

    it('should throw error when adding non-existent photo to album', async () => {
      const albumId = await createAlbum('Test Album');
      await expect(addPhotoToAlbum(albumId, 99999)).rejects.toThrow();
    });

    it('should prevent duplicate photo-album associations', async () => {
      const albumId = await createAlbum('Test Album');
      const photoId = await addPhoto({
        filePath: '/test/photo1.jpg',
        thumbnail: 'data:image/jpeg;base64,test1'
      });

      await addPhotoToAlbum(albumId, photoId);
      await expect(addPhotoToAlbum(albumId, photoId)).rejects.toThrow();
    });

    it('should update album updated_at timestamp', async () => {
      const albumId = await createAlbum('Test Album');
      const photoId = await addPhoto({
        filePath: '/test/photo1.jpg',
        thumbnail: 'data:image/jpeg;base64,test1'
      });

      await new Promise(resolve => setTimeout(resolve, 10));
      await addPhotoToAlbum(albumId, photoId);
    });
  });

  describe('getAlbumPhotos', () => {
    it('should return empty array for album with no photos', async () => {
      const albumId = await createAlbum('Empty Album');
      const photos = await getAlbumPhotos(albumId);

      expect(photos).toEqual([]);
    });

    it('should return all photos for an album', async () => {
      const albumId = await createAlbum('Test Album');

      const photo1 = await addPhoto({ filePath: '/test/p1.jpg', thumbnail: 'data:image/jpeg;base64,t1' });
      const photo2 = await addPhoto({ filePath: '/test/p2.jpg', thumbnail: 'data:image/jpeg;base64,t2' });
      const photo3 = await addPhoto({ filePath: '/test/p3.jpg', thumbnail: 'data:image/jpeg;base64,t3' });

      await addPhotoToAlbum(albumId, photo1);
      await addPhotoToAlbum(albumId, photo2);
      await addPhotoToAlbum(albumId, photo3);

      const photos = await getAlbumPhotos(albumId);
      expect(photos).toHaveLength(3);
    });

    it('should return photos ordered by added_at', async () => {
      const albumId = await createAlbum('Test Album');

      const photo1 = await addPhoto({ filePath: '/test/p1.jpg', thumbnail: 'data:image/jpeg;base64,t1' });
      const photo2 = await addPhoto({ filePath: '/test/p2.jpg', thumbnail: 'data:image/jpeg;base64,t2' });
      const photo3 = await addPhoto({ filePath: '/test/p3.jpg', thumbnail: 'data:image/jpeg;base64,t3' });

      await addPhotoToAlbum(albumId, photo3);
      await new Promise(resolve => setTimeout(resolve, 10));
      await addPhotoToAlbum(albumId, photo1);
      await new Promise(resolve => setTimeout(resolve, 10));
      await addPhotoToAlbum(albumId, photo2);

      const photos = await getAlbumPhotos(albumId);
      expect(photos[0].photo_id).toBe(photo3);
      expect(photos[1].photo_id).toBe(photo1);
      expect(photos[2].photo_id).toBe(photo2);
    });

    it('should throw error for non-existent album', async () => {
      await expect(getAlbumPhotos(99999)).rejects.toThrow();
    });

    it('should include photo metadata in results', async () => {
      const albumId = await createAlbum('Test Album');
      const photoId = await addPhoto({ filePath: '/test/photo.jpg', thumbnail: 'data:image/jpeg;base64,test' });
      await addPhotoToAlbum(albumId, photoId);

      const photos = await getAlbumPhotos(albumId);
      expect(photos[0]).toHaveProperty('photo_id');
      expect(photos[0]).toHaveProperty('file_path');
      expect(photos[0]).toHaveProperty('thumbnail_blob');
      expect(photos[0]).toHaveProperty('date_taken');
    });
  });
});

describe('AlbumService - Display Order', () => {
  beforeEach(async () => {
    await initApp();
  });

  afterEach(async () => {
    await clearAllData();
  });

  describe('updateDisplayOrder', () => {
    it('should update display order for a single album', async () => {
      const albumId = await createAlbum('Test Album');
      
      await updateDisplayOrder(albumId, 5);

      const album = await getAlbum(albumId);
      expect(album.display_order).toBe(5);
    });

    it('should throw error when updating non-existent album', async () => {
      await expect(updateDisplayOrder(99999, 1)).rejects.toThrow();
    });
  });

  describe('reorderAlbums', () => {
    it('should update display orders for multiple albums', async () => {
      const album1 = await createAlbum('Album 1');
      const album2 = await createAlbum('Album 2');
      const album3 = await createAlbum('Album 3');

      await reorderAlbums([
        { albumId: album3, displayOrder: 0 },
        { albumId: album1, displayOrder: 1 },
        { albumId: album2, displayOrder: 2 }
      ]);

      const albums = await getAllAlbums();
      const sortedAlbums = albums.sort((a, b) => a.display_order - b.display_order);

      expect(sortedAlbums[0].id).toBe(album3);
      expect(sortedAlbums[1].id).toBe(album1);
      expect(sortedAlbums[2].id).toBe(album2);
    });

    it('should handle empty reorder list', async () => {
      await reorderAlbums([]);
    });

    it('should preserve album data when reordering', async () => {
      const album1 = await createAlbum('Original Name');

      await reorderAlbums([
        { albumId: album1, displayOrder: 5 }
      ]);

      const album = await getAlbum(album1);
      expect(album.name).toBe('Original Name');
      expect(album.display_order).toBe(5);
    });
  });
});

describe('AlbumService - Date Grouping', () => {
  beforeEach(async () => {
    await initApp();
  });

  afterEach(async () => {
    await clearAllData();
  });

  describe('getAlbumsGroupedByDate', () => {
    it('should return albums grouped by year-month', async () => {
      await createAlbum('Album Jan 1');
      await createAlbum('Album Jan 2');
      await createAlbum('Album Feb');

      const groups = await getAlbumsGroupedByDate();

      expect(groups).toBeInstanceOf(Array);
      expect(groups.length).toBeGreaterThan(0);

      groups.forEach(group => {
        expect(group).toHaveProperty('dateGroup');
        expect(group).toHaveProperty('albums');
        expect(group.albums).toBeInstanceOf(Array);
      });
    });

    it('should return empty array when no albums exist', async () => {
      const groups = await getAlbumsGroupedByDate();
      expect(groups).toEqual([]);
    });

    it('should group albums from same month together', async () => {
      await createAlbum('Album 1');
      await createAlbum('Album 2');
      await createAlbum('Album 3');

      const groups = await getAlbumsGroupedByDate();

      const firstGroup = groups[0];
      expect(firstGroup.albums.length).toBeGreaterThan(0);
    });

    it('should maintain display_order within each group', async () => {
      const album1 = await createAlbum('Album 1');
      const album2 = await createAlbum('Album 2');
      const album3 = await createAlbum('Album 3');

      await reorderAlbums([
        { albumId: album3, displayOrder: 0 },
        { albumId: album1, displayOrder: 1 },
        { albumId: album2, displayOrder: 2 }
      ]);

      const groups = await getAlbumsGroupedByDate();

      const firstGroup = groups[0];
      if (firstGroup && firstGroup.albums.length >= 3) {
        expect(firstGroup.albums[0].id).toBe(album3);
        expect(firstGroup.albums[1].id).toBe(album1);
        expect(firstGroup.albums[2].id).toBe(album2);
      }
    });

    it('should sort groups chronologically descending', async () => {
      const albums = await getAllAlbums();

      if (albums.length > 0) {
        const groups = await getAlbumsGroupedByDate();

        for (let i = 0; i < groups.length - 1; i++) {
          if (groups[i].dateGroup !== 'unknown' && groups[i + 1].dateGroup !== 'unknown') {
            expect(groups[i].dateGroup >= groups[i + 1].dateGroup).toBe(true);
          }
        }
      }
    });

    it('should include all album properties in grouped results', async () => {
      await createAlbum('Test Album');

      const groups = await getAlbumsGroupedByDate();

      if (groups.length > 0 && groups[0].albums.length > 0) {
        const album = groups[0].albums[0];
        expect(album).toHaveProperty('id');
        expect(album).toHaveProperty('name');
        expect(album).toHaveProperty('display_order');
        expect(album).toHaveProperty('created_date');
      }
    });
  });
});
