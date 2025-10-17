# Research: Photo Album Organizer

**Created**: 2025-10-16
**Phase**: 0 (Technical Research & Decisions)

## Overview

This document captures technical research and decisions for building a browser-based photo album organizer with minimal dependencies.

## Decision 1: SQLite Storage Strategy

**Context**: Need local database for album/photo metadata without server infrastructure

**Decision**: Use sql.js (SQLite compiled to WebAssembly)

**Rationale**:
- Pure browser solution, no Electron or Node.js required
- Runs entirely client-side meeting offline requirement
- Full SQL capability for complex queries (date grouping, ordering)
- Smaller footprint than alternative databases
- Persistence via IndexedDB (save/load SQLite database file)

**Alternatives Considered**:
- IndexedDB directly: Rejected - query complexity for date grouping would be cumbersome
- LocalStorage: Rejected - 5-10MB limit insufficient for 1000+ albums
- better-sqlite3: Rejected - requires Node.js/Electron, adds deployment complexity

**Implementation Notes**:
- Load sql.js WASM module on app initialization
- Create/open SQLite database in memory
- Persist database to IndexedDB on changes (debounced)
- Restore database from IndexedDB on app load

## Decision 2: Image Handling with File System Access API

**Context**: Users need to add photos without uploading files

**Decision**: Use File System Access API for modern browsers

**Rationale**:
- Allows persistent file handles without copying images
- User grants permission once, handles remain valid
- Zero storage overhead (no image duplication)
- Works offline by design
- Supported in Chrome 86+, Edge 86+, Safari 15.2+

**Alternatives Considered**:
- File input + ObjectURL: Rejected - URLs expire on page refresh, no persistence
- Copy files to IndexedDB: Rejected - massive storage consumption, violates "no upload" requirement
- Drag-drop only: Rejected - insufficient for persistent access

**Implementation Notes**:
- Request file handles via showOpenFilePicker()
- Store serialized file handles in SQLite (path reference)
- Request permission on app load to re-access files
- Generate thumbnails in-memory using Canvas API
- Cache thumbnails in IndexedDB for performance

## Decision 3: EXIF Metadata Extraction

**Context**: Need to extract date information from photos

**Decision**: Use exifr library (minimal EXIF reader)

**Rationale**:
- Lightweight (10-15KB gzipped)
- Fast parsing, works with File/Blob objects
- Supports all major image formats (JPEG, PNG, HEIC)
- Tree-shakeable (only include needed parsers)

**Alternatives Considered**:
- exif-js: Rejected - unmaintained, larger bundle
- piexifjs: Rejected - write-focused, unnecessary features
- Custom parser: Rejected - complex specification, error-prone

**Implementation Notes**:
- Extract DateTimeOriginal, DateTime, CreateDate fields
- Fallback to file.lastModified if no EXIF date
- Parse dates to ISO format for SQLite storage

## Decision 4: Drag-and-Drop Implementation

**Context**: Users need to reorder albums via drag-and-drop

**Decision**: Native HTML5 Drag and Drop API with custom visual feedback

**Rationale**:
- No library needed, built into browsers
- Accessible with keyboard fallbacks
- Meets <100ms response time requirement
- Full control over visual feedback

**Alternatives Considered**:
- SortableJS: Rejected - adds 30KB+ dependency, violates minimal library requirement
- Pointer events manual implementation: Rejected - complex touch handling
- CSS Grid reorder: Rejected - no drag feedback

**Implementation Notes**:
- Use draggable attribute on album elements
- Implement ondragstart, ondragover, ondrop handlers
- Visual feedback via CSS classes during drag
- Update album.order field in SQLite on drop
- Debounce database writes during rapid reordering

## Decision 5: Virtual Scrolling for Large Photo Grids

**Context**: Albums may contain 10,000+ photos, DOM performance concern

**Decision**: Implement simple virtual scrolling (windowing)

**Rationale**:
- Only render visible + buffer photos in viewport
- Dramatically reduces DOM nodes (thousands → hundreds)
- Meets performance requirement (100+ photos smooth)
- Can implement in <100 LOC vanilla JS

**Alternatives Considered**:
- React Window/react-virtualized: Rejected - requires React framework
- Render all photos: Rejected - DOM performance degrades >500 elements
- Pagination: Rejected - poor UX for browsing, requires clicks

**Implementation Notes**:
- Calculate visible range based on scroll position
- Render visible tiles + 10 rows buffer (top/bottom)
- Use CSS Grid with fixed row height for layout
- Recalculate on scroll (throttled to 16ms/60fps)
- Maintain scroll position on data updates

## Decision 6: Thumbnail Generation Strategy

**Context**: Need fast thumbnail loading without backend processing

**Decision**: Generate thumbnails on-demand using Canvas API, cache in IndexedDB

**Rationale**:
- Canvas.drawImage() can resize images efficiently
- Cache prevents regeneration (IndexedDB blob storage)
- Lazy loading: generate thumbnails as user scrolls
- Keeps bundle small, no image processing library

**Alternatives Considered**:
- Store full images in IndexedDB: Rejected - massive storage waste
- Always load full images: Rejected - performance impact for grid view
- WebP encoding via OffscreenCanvas: Considered but adds complexity, defer to optimization phase

**Implementation Notes**:
- Target thumbnail size: 300x300px (CSS Grid tile size)
- Generate thumbnails at 2x resolution for retina displays (600x600px)
- Store as JPEG blob in IndexedDB (key: file_path_hash + size)
- Lazy generation: create thumbnail when tile enters viewport
- Cache eviction: LRU if IndexedDB exceeds 500MB

## Decision 7: Date Grouping Logic

**Context**: Albums need automatic grouping by date

**Decision**: Month-year grouping (e.g., "October 2025") using SQL GROUP BY

**Rationale**:
- Month granularity balances organization vs. too many groups
- SQL query can efficiently group albums by strftime('%Y-%m', date)
- Supports both photo dates and album creation dates
- Locale-aware month names via JavaScript Intl.DateTimeFormat

**Alternatives Considered**:
- Day grouping: Rejected - too granular, many single-album groups
- Year only: Rejected - too coarse for active photographers
- Custom date ranges: Deferred - can add as future enhancement

**Implementation Notes**:
- Store album representative date (earliest photo date in album)
- Query: SELECT strftime('%Y-%m', album_date) as month, * FROM albums GROUP BY month
- Display with Intl.DateTimeFormat: { month: 'long', year: 'numeric' }
- Allow toggle between grouped/flat view

## Performance Benchmarks

**Established Targets** (from constitution and spec):
- Album creation: < 30 seconds
- Photo grid rendering: 100+ photos without lag (60fps)
- Drag-drop response: < 100ms
- Database query: < 100ms
- Initial bundle load: < 250KB gzipped
- UI interactions: < 200ms p95

**Validation Strategy**:
- Use Performance API to measure critical operations
- Lighthouse CI for bundle size and load time
- Playwright tests with performance assertions
- Chrome DevTools Performance profiler for frame drops

## Accessibility Considerations

**WCAG 2.1 Level AA Requirements**:
- Keyboard navigation: Tab through albums, Enter to open, Arrow keys to navigate grid
- Focus indicators: Visible focus rings on all interactive elements
- ARIA labels: aria-label for album cards, aria-describedby for photo counts
- Color contrast: Minimum 4.5:1 for text, 3:1 for UI components
- Screen reader support: Announce drag-drop state changes

**Testing Strategy**:
- axe-core automated testing in integration tests
- Manual testing with NVDA/JAWS screen readers
- Keyboard-only navigation validation
- Color contrast checker integration

## Summary of Technical Stack

**Runtime**:
- JavaScript ES2022 (modules, async/await, optional chaining)
- HTML5 (semantic elements, File System Access API)
- CSS3 (Grid, Flexbox, custom properties)

**Build Tools**:
- Vite 5.x (dev server + production build)
- Vitest (unit testing)
- Playwright (E2E testing)

**Dependencies** (total: 3):
1. sql.js (~500KB WASM, lazy-loaded)
2. exifr (~15KB gzipped, tree-shaken)
3. Vite (dev-only, not in bundle)

**Browser APIs**:
- File System Access API (file handles)
- IndexedDB (database + thumbnail persistence)
- Canvas API (thumbnail generation)
- Drag and Drop API (album reordering)
- IntersectionObserver (lazy loading)
- ResizeObserver (responsive grid)

**Bundle Size Projection**:
- Application code: ~50KB (minified + gzipped)
- sql.js WASM: ~500KB (lazy-loaded, not in initial bundle)
- exifr: ~15KB (gzipped)
- **Initial load: ~65KB** ✅ Well under 250KB target

**All technical decisions align with PhotoAlbum Constitution v1.0.0 principles.**
