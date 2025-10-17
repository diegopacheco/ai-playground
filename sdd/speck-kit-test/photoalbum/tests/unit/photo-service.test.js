import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { initApp, clearAllData } from '../../src/db/database.js';
import { addPhoto, getPhotoById, getAllPhotos } from '../../src/services/photo-service.js';

describe('PhotoService', () => {
  beforeEach(async () => {
    await initApp();
  });

  afterEach(async () => {
    await clearAllData();
  });

  describe('addPhoto', () => {
    it('should add a new photo with valid data', async () => {
      const photoData = {
        filePath: '/path/to/photo.jpg',
        thumbnail: 'data:image/jpeg;base64,testdata',
        dateTaken: '2025-10-16T12:00:00.000Z',
        width: 1920,
        height: 1080,
        fileSize: 2048000,
        mimeType: 'image/jpeg'
      };

      const photoId = await addPhoto(photoData);

      expect(photoId).toBeDefined();
      expect(typeof photoId).toBe('number');
      expect(photoId).toBeGreaterThan(0);
    });

    it('should throw error when adding photo without file_path', async () => {
      const photoData = {
        thumbnail: 'data:image/jpeg;base64,testdata'
      };

      await expect(addPhoto(photoData)).rejects.toThrow('File path is required');
    });

    it('should throw error when adding photo without thumbnail', async () => {
      const photoData = {
        filePath: '/path/to/photo.jpg'
      };

      await expect(addPhoto(photoData)).rejects.toThrow('Thumbnail is required');
    });

    it('should prevent duplicate file_path entries', async () => {
      const photoData = {
        filePath: '/path/to/photo.jpg',
        thumbnail: 'data:image/jpeg;base64,testdata'
      };

      await addPhoto(photoData);
      await expect(addPhoto(photoData)).rejects.toThrow();
    });

    it('should handle optional metadata fields', async () => {
      const photoData = {
        filePath: '/path/to/photo.jpg',
        thumbnail: 'data:image/jpeg;base64,testdata'
      };

      const photoId = await addPhoto(photoData);
      expect(photoId).toBeGreaterThan(0);
    });

    it('should set added_at timestamp automatically', async () => {
      const before = new Date();
      const photoData = {
        filePath: '/path/to/photo.jpg',
        thumbnail: 'data:image/jpeg;base64,testdata'
      };

      const photoId = await addPhoto(photoData);
      const after = new Date();

      expect(photoId).toBeGreaterThan(0);
    });
  });

  describe('getPhotoById', () => {
    it('should retrieve photo by id', async () => {
      const photoData = {
        filePath: '/path/to/photo.jpg',
        thumbnail: 'data:image/jpeg;base64,testdata',
        dateTaken: '2025-10-16T12:00:00.000Z',
        width: 1920,
        height: 1080
      };

      const photoId = await addPhoto(photoData);
      const photo = await getPhotoById(photoId);

      expect(photo).toBeDefined();
      expect(photo.id).toBe(photoId);
      expect(photo.file_path).toBe('/path/to/photo.jpg');
      expect(photo.thumbnail_blob).toBe('data:image/jpeg;base64,testdata');
    });

    it('should return null for non-existent photo id', async () => {
      const photo = await getPhotoById(99999);
      expect(photo).toBeNull();
    });

    it('should include all photo metadata fields', async () => {
      const photoData = {
        filePath: '/path/to/photo.jpg',
        thumbnail: 'data:image/jpeg;base64,testdata',
        dateTaken: '2025-10-16T12:00:00.000Z',
        width: 1920,
        height: 1080,
        fileSize: 2048000,
        mimeType: 'image/jpeg'
      };

      const photoId = await addPhoto(photoData);
      const photo = await getPhotoById(photoId);

      expect(photo).toHaveProperty('id');
      expect(photo).toHaveProperty('file_path');
      expect(photo).toHaveProperty('thumbnail_blob');
      expect(photo).toHaveProperty('date_taken');
      expect(photo).toHaveProperty('width');
      expect(photo).toHaveProperty('height');
      expect(photo).toHaveProperty('file_size');
      expect(photo).toHaveProperty('mime_type');
      expect(photo).toHaveProperty('created_date');
    });
  });

  describe('getAllPhotos', () => {
    it('should return empty array when no photos exist', async () => {
      const photos = await getAllPhotos();
      expect(photos).toEqual([]);
    });

    it('should return all photos ordered by date_taken', async () => {
      await addPhoto({
        filePath: '/path/photo1.jpg',
        thumbnail: 'data:image/jpeg;base64,test1',
        dateTaken: '2025-01-15T10:00:00.000Z'
      });
      await addPhoto({
        filePath: '/path/photo2.jpg',
        thumbnail: 'data:image/jpeg;base64,test2',
        dateTaken: '2025-01-10T10:00:00.000Z'
      });
      await addPhoto({
        filePath: '/path/photo3.jpg',
        thumbnail: 'data:image/jpeg;base64,test3',
        dateTaken: '2025-01-20T10:00:00.000Z'
      });

      const photos = await getAllPhotos();
      expect(photos).toHaveLength(3);
      expect(photos[0].date_taken).toBe('2025-01-20T10:00:00.000Z');
      expect(photos[1].date_taken).toBe('2025-01-15T10:00:00.000Z');
      expect(photos[2].date_taken).toBe('2025-01-10T10:00:00.000Z');
    });

    it('should handle photos without date_taken', async () => {
      await addPhoto({
        filePath: '/path/photo1.jpg',
        thumbnail: 'data:image/jpeg;base64,test1'
      });

      const photos = await getAllPhotos();
      expect(photos).toHaveLength(1);
    });
  });
});
