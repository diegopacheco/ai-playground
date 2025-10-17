import { describe, it, expect, beforeEach } from 'vitest';
import { generateThumbnail, blobToBase64, base64ToBlob, createThumbnailURL, revokeThumbnailURL } from '../../src/utils/thumbnail-generator.js';

describe('Thumbnail Generator', () => {
  let mockCanvas;
  let mockContext;

  beforeEach(() => {
    mockContext = {
      drawImage: () => {}
    };

    mockCanvas = {
      getContext: () => mockContext,
      toBlob: (callback) => {
        const mockBlob = new Blob(['mock image data'], { type: 'image/jpeg' });
        callback(mockBlob);
      },
      width: 0,
      height: 0
    };

    global.document = {
      createElement: (tag) => {
        if (tag === 'canvas') return mockCanvas;
        return {};
      }
    };
  });

  it('should convert blob to base64', async () => {
    const blob = new Blob(['test data'], { type: 'text/plain' });
    const base64 = await blobToBase64(blob);

    expect(base64).toBeDefined();
    expect(base64).toContain('data:');
    expect(base64).toContain('base64,');
  });

  it('should convert base64 to blob', () => {
    const base64 = 'data:text/plain;base64,dGVzdCBkYXRh';
    const blob = base64ToBlob(base64);

    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('text/plain');
  });

  it('should create thumbnail URL from blob', () => {
    const blob = new Blob(['test'], { type: 'image/jpeg' });
    const url = createThumbnailURL(blob);

    expect(url).toBeDefined();
    expect(typeof url).toBe('string');
    expect(url).toContain('blob:');
  });

  it('should handle base64 to blob conversion errors', () => {
    expect(() => {
      base64ToBlob('invalid-base64-string');
    }).toThrow();
  });

  it('should revoke thumbnail URL without errors', () => {
    const blob = new Blob(['test'], { type: 'image/jpeg' });
    const url = createThumbnailURL(blob);

    expect(() => {
      revokeThumbnailURL(url);
    }).not.toThrow();
  });
});
