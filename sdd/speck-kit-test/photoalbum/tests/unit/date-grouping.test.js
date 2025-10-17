import { describe, it, expect } from 'vitest';
import { computeDateGroups, formatDateGroup, sortAlbumsByDate } from '../../src/utils/date-grouping.js';

describe('Date Grouping Utilities', () => {
  describe('computeDateGroups', () => {
    it('should group albums by year-month', () => {
      const albums = [
        { id: 1, name: 'Album 1', created_date: '2024-01-15T10:00:00Z', display_order: 0 },
        { id: 2, name: 'Album 2', created_date: '2024-01-20T10:00:00Z', display_order: 1 },
        { id: 3, name: 'Album 3', created_date: '2024-02-10T10:00:00Z', display_order: 0 },
      ];

      const groups = computeDateGroups(albums);

      expect(groups).toHaveLength(2);
      expect(groups[0].dateGroup).toBe('2024-02');
      expect(groups[0].albums).toHaveLength(1);
      expect(groups[1].dateGroup).toBe('2024-01');
      expect(groups[1].albums).toHaveLength(2);
    });

    it('should handle albums from different years', () => {
      const albums = [
        { id: 1, name: 'Album 1', created_date: '2023-12-15T10:00:00Z', display_order: 0 },
        { id: 2, name: 'Album 2', created_date: '2024-01-15T10:00:00Z', display_order: 0 },
      ];

      const groups = computeDateGroups(albums);

      expect(groups).toHaveLength(2);
      expect(groups[0].dateGroup).toBe('2024-01');
      expect(groups[1].dateGroup).toBe('2023-12');
    });

    it('should return empty array for empty albums list', () => {
      const groups = computeDateGroups([]);
      expect(groups).toEqual([]);
    });

    it('should handle albums without created_date', () => {
      const albums = [
        { id: 1, name: 'Album 1', created_date: '2024-01-15T10:00:00Z' },
        { id: 2, name: 'Album 2', created_date: null },
      ];

      const groups = computeDateGroups(albums);

      expect(groups).toHaveLength(2);
      expect(groups.find(g => g.dateGroup === '2024-01')).toBeDefined();
      expect(groups.find(g => g.dateGroup === 'unknown')).toBeDefined();
    });

    it('should sort groups chronologically descending', () => {
      const albums = [
        { id: 1, name: 'Album 1', created_date: '2024-01-15T10:00:00Z' },
        { id: 2, name: 'Album 2', created_date: '2024-03-15T10:00:00Z' },
        { id: 3, name: 'Album 3', created_date: '2024-02-15T10:00:00Z' },
      ];

      const groups = computeDateGroups(albums);

      expect(groups[0].dateGroup).toBe('2024-03');
      expect(groups[1].dateGroup).toBe('2024-02');
      expect(groups[2].dateGroup).toBe('2024-01');
    });

    it('should maintain album display_order within groups', () => {
      const albums = [
        { id: 1, name: 'Album 1', created_date: '2024-01-15T10:00:00Z', display_order: 1 },
        { id: 2, name: 'Album 2', created_date: '2024-01-20T10:00:00Z', display_order: 0 },
      ];

      const groups = computeDateGroups(albums);

      expect(groups[0].albums[0].display_order).toBe(0);
      expect(groups[0].albums[1].display_order).toBe(1);
    });
  });

  describe('formatDateGroup', () => {
    it('should format year-month as readable text', () => {
      expect(formatDateGroup('2024-01')).toBe('January 2024');
      expect(formatDateGroup('2024-12')).toBe('December 2024');
    });

    it('should handle unknown date group', () => {
      expect(formatDateGroup('unknown')).toBe('Unknown Date');
    });

    it('should handle invalid format', () => {
      expect(formatDateGroup('invalid')).toBe('Unknown Date');
    });
  });

  describe('sortAlbumsByDate', () => {
    it('should sort albums by created_date descending', () => {
      const albums = [
        { id: 1, created_date: '2024-01-15T10:00:00Z' },
        { id: 2, created_date: '2024-03-15T10:00:00Z' },
        { id: 3, created_date: '2024-02-15T10:00:00Z' },
      ];

      const sorted = sortAlbumsByDate(albums);

      expect(sorted[0].id).toBe(2);
      expect(sorted[1].id).toBe(3);
      expect(sorted[2].id).toBe(1);
    });

    it('should handle null dates by placing them last', () => {
      const albums = [
        { id: 1, created_date: '2024-01-15T10:00:00Z' },
        { id: 2, created_date: null },
        { id: 3, created_date: '2024-02-15T10:00:00Z' },
      ];

      const sorted = sortAlbumsByDate(albums);

      expect(sorted[0].id).toBe(3);
      expect(sorted[1].id).toBe(1);
      expect(sorted[2].id).toBe(2);
    });
  });
});
