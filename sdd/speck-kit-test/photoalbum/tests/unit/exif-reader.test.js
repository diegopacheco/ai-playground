import { describe, it, expect } from 'vitest';
import { extractPhotoMetadata, formatDateForDisplay } from '../../src/utils/exif-reader.js';

describe('EXIF Reader', () => {
  it('should extract metadata from image file', async () => {
    const mockFile = new File([''], 'test.jpg', {
      type: 'image/jpeg',
      lastModified: Date.now()
    });

    const metadata = await extractPhotoMetadata(mockFile);

    expect(metadata).toBeDefined();
    expect(metadata).toHaveProperty('dateTaken');
    expect(metadata).toHaveProperty('width');
    expect(metadata).toHaveProperty('height');
    expect(metadata).toHaveProperty('fileSize');
    expect(metadata).toHaveProperty('mimeType');
    expect(metadata).toHaveProperty('fileName');
  });

  it('should use file lastModified as fallback date', async () => {
    const lastModified = new Date('2025-01-01').getTime();
    const mockFile = new File([''], 'test.jpg', {
      type: 'image/jpeg',
      lastModified
    });

    const metadata = await extractPhotoMetadata(mockFile);

    expect(metadata.dateTaken).toBeDefined();
    expect(new Date(metadata.dateTaken).getTime()).toBeGreaterThan(0);
  });

  it('should handle non-image files gracefully', async () => {
    const mockFile = new File([''], 'test.txt', {
      type: 'text/plain',
      lastModified: Date.now()
    });

    const metadata = await extractPhotoMetadata(mockFile);

    expect(metadata).toBeDefined();
    expect(metadata.dateTaken).toBeDefined();
  });

  it('should format date for display', () => {
    const isoDate = '2025-10-16T12:00:00.000Z';
    const formatted = formatDateForDisplay(isoDate);

    expect(formatted).toBeDefined();
    expect(formatted).not.toBe('Unknown Date');
    expect(formatted).toContain('2025');
  });

  it('should handle null date gracefully', () => {
    const formatted = formatDateForDisplay(null);
    expect(formatted).toBe('Unknown Date');
  });

  it('should handle invalid date string', () => {
    const formatted = formatDateForDisplay('invalid-date');
    expect(formatted).toBe('Unknown Date');
  });

  it('should extract file size correctly', async () => {
    const fileSize = 1024 * 100;
    const mockFile = new File([new ArrayBuffer(fileSize)], 'test.jpg', {
      type: 'image/jpeg'
    });

    const metadata = await extractPhotoMetadata(mockFile);
    expect(metadata.fileSize).toBe(fileSize);
  });

  it('should extract MIME type correctly', async () => {
    const mockFile = new File([''], 'test.png', {
      type: 'image/png'
    });

    const metadata = await extractPhotoMetadata(mockFile);
    expect(metadata.mimeType).toBe('image/png');
  });
});
