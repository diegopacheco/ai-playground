# Quickstart Guide: Photo Album Organizer

**Created**: 2025-10-16
**Phase**: 1 (Design & Contracts)
**Purpose**: User guide for testing the photo album organizer application

## Prerequisites

### System Requirements
- Modern web browser (Chrome 100+, Firefox 100+, or Safari 15.2+)
- Desktop or laptop computer (mobile support not included in MVP)
- Photos stored locally on your device

### Browser Permissions
The application requires the following browser permissions:
- **File System Access**: To read and display your photos
- **Storage**: To save album organization data locally

## Installation

### Option 1: Development Mode
```bash
cd photoalbum
npm install
npm run dev
```
Application will open at `http://localhost:5173`

### Option 2: Production Build
```bash
cd photoalbum
npm install
npm run build
npm run preview
```
Application will open at `http://localhost:4173`

## First-Time Setup

### Step 1: Launch Application
1. Open the application URL in your browser
2. You'll see an empty main page with "Create Album" button

### Step 2: Grant File Access Permission
1. The first time you select photos, your browser will ask for permission
2. Click "Allow" to grant the application access to your photos
3. This permission persists across sessions

## User Scenarios

### Scenario 1: Create Your First Album (P1 - MVP)

**Objective**: Create a new album and add photos

**Steps**:
1. Click the **"Create Album"** button on the main page
2. Enter an album name (e.g., "Vacation 2025")
3. Click **"Add Photos"** button
4. In the file picker dialog, select one or more photos from your device
   - Hold `Ctrl` (Windows/Linux) or `Cmd` (Mac) to select multiple photos
   - Or select a range with `Shift`
5. Click **"Open"**
6. Photos will appear in a tile grid within the album

**Expected Results**:
- New album appears on the main page
- Album shows a thumbnail preview (first photo)
- Album displays photo count
- Opening the album shows all photos in a tile grid

**Test Validation**:
- ✅ Album creation takes < 30 seconds
- ✅ Photos appear immediately after selection
- ✅ Tile grid displays smoothly (60fps)

### Scenario 2: View Album Photos

**Objective**: View photos within an album in tile layout

**Steps**:
1. From the main page, click on an existing album
2. Album opens showing all photos in a grid layout
3. Scroll through the photos (if more than fit on screen)

**Expected Results**:
- Photos display in a responsive tile grid (3-5 columns depending on screen width)
- Scrolling is smooth with no lag
- Each photo tile shows thumbnail preview
- Click "Back" or browser back button to return to main page

**Test Validation**:
- ✅ 100+ photos display without performance issues
- ✅ Virtual scrolling works for large albums
- ✅ Thumbnails load progressively as you scroll

### Scenario 3: Add More Photos to Existing Album

**Objective**: Expand an album by adding additional photos

**Steps**:
1. Open an existing album
2. Click **"Add Photos"** button
3. Select additional photos from your device
4. Click **"Open"**

**Expected Results**:
- New photos appear in the tile grid immediately
- Photos are added to the end of the album
- Album photo count updates on main page

**Test Validation**:
- ✅ New photos appear without page refresh
- ✅ Album thumbnail may update if it was empty
- ✅ Photo count increments correctly

### Scenario 4: Reorder Albums (P2)

**Objective**: Reorganize albums on the main page via drag-and-drop

**Steps**:
1. On the main page with multiple albums (create 3+ albums for testing)
2. Click and hold on an album card
3. Drag the album to a new position
4. Release to drop in the new location
5. Refresh the page to verify persistence

**Expected Results**:
- Album moves smoothly during drag
- Visual feedback shows drop position (gap or indicator)
- Album stays in new position after drop
- Order persists after page refresh

**Test Validation**:
- ✅ Drag-drop responds within 100ms
- ✅ Smooth animation during drag
- ✅ Order saves to database automatically
- ✅ Order restored correctly on page load

**Keyboard Alternative**:
- Tab to focus an album
- Press `Space` to enter move mode
- Use `Arrow keys` to change position
- Press `Enter` to confirm new position

### Scenario 5: Date-Based Album Grouping (P3)

**Objective**: View albums grouped by date (month/year)

**Setup**: Create albums with photos from different dates

**Steps**:
1. Create albums containing photos from different months (use photo metadata)
2. On the main page, click **"Group by Date"** toggle
3. Albums reorganize into date sections

**Expected Results**:
- Albums grouped under month/year headers (e.g., "October 2025", "September 2025")
- Sections ordered chronologically (newest first)
- Each section shows albums from that month
- Toggle off returns to flat view

**Test Validation**:
- ✅ Date extraction from photo EXIF metadata works
- ✅ Albums without dates appear in "Unknown Date" section
- ✅ Grouping is 95%+ accurate based on photo dates
- ✅ Drag-drop still works within date groups

### Scenario 6: Remove Photos from Album

**Objective**: Delete photos from an album

**Steps**:
1. Open an album with photos
2. Hover over a photo tile
3. Click the **"Remove"** (×) button in the corner
4. Confirm deletion if prompted

**Expected Results**:
- Photo disappears from the album immediately
- Album photo count decrements
- Photo is NOT deleted from your file system (only removed from album)
- Same photo may still appear in other albums

**Test Validation**:
- ✅ Remove operation is instant
- ✅ No data loss (original file untouched)
- ✅ Album updates correctly

### Scenario 7: Delete Empty Album

**Objective**: Remove an album that has no photos

**Steps**:
1. Create a new album without adding photos (or remove all photos from existing album)
2. On the main page, click **"Delete"** button on the empty album
3. Confirm deletion

**Expected Results**:
- Album disappears from main page
- Album data removed from database
- No photos are deleted from file system

**Test Validation**:
- ✅ Empty albums can be deleted
- ✅ Deletion is permanent (no undo in MVP)

## Edge Cases to Test

### Edge Case 1: Photos Without Date Metadata
**Setup**: Add photos that have no EXIF date (e.g., screenshots, downloaded images)

**Expected Behavior**:
- App uses file creation date as fallback
- Photos appear in "Unknown Date" group if date grouping is enabled
- No errors or crashes

### Edge Case 2: Duplicate Photo in Multiple Albums
**Setup**: Add the same photo file to two different albums

**Expected Behavior**:
- Photo appears in both albums
- Removing from one album doesn't affect the other
- Storage efficient (no file duplication, only metadata)

### Edge Case 3: Very Large Album (1000+ Photos)
**Setup**: Create an album with 1000+ photos

**Expected Behavior**:
- Virtual scrolling activates automatically
- Smooth 60fps scrolling
- Thumbnails load lazily as you scroll
- No memory issues or crashes

### Edge Case 4: Reordering with Date Grouping Enabled
**Setup**: Enable date grouping, then try to drag albums

**Expected Behavior**:
- Albums can only be reordered within their date group
- Cannot drag album into different date group
- Visual feedback prevents invalid drops

### Edge Case 5: Browser Permission Denied
**Setup**: Deny file access permission when prompted

**Expected Behavior**:
- Clear error message: "File access required to add photos"
- Instructions on how to re-enable permission in browser settings
- Application doesn't crash

## Performance Benchmarks

Run these tests to validate performance requirements:

### Test 1: Album Creation Speed
1. Time from clicking "Create Album" to album appearing on main page
2. **Target**: < 30 seconds

### Test 2: Photo Grid Rendering
1. Open album with 100+ photos
2. Use browser DevTools Performance tab
3. Check frame rate while scrolling
4. **Target**: 60fps (16.7ms per frame)

### Test 3: Drag-Drop Responsiveness
1. Start dragging an album
2. Measure time until visual feedback appears
3. **Target**: < 100ms

### Test 4: Database Query Performance
1. Open browser DevTools Console
2. Check database query times (logged in dev mode)
3. **Target**: < 100ms per query

### Test 5: Initial Bundle Load
1. Open Network tab in DevTools
2. Hard refresh page (Ctrl+Shift+R)
3. Check total JavaScript size (gzipped)
4. **Target**: < 250KB

## Accessibility Testing

### Keyboard Navigation Test
1. Use only keyboard (no mouse)
2. Tab through all interactive elements
3. Enter to activate buttons
4. Arrow keys to navigate

**Expected**: All functionality accessible via keyboard

### Screen Reader Test
1. Enable screen reader (NVDA on Windows, VoiceOver on Mac)
2. Navigate through the application
3. Verify announcements for all actions

**Expected**: All UI elements have proper labels and announcements

### Color Contrast Test
1. Use browser extension (e.g., axe DevTools)
2. Run accessibility audit
3. Check contrast ratios

**Expected**: All text meets WCAG 2.1 AA contrast requirements (4.5:1)

## Troubleshooting

### Issue: "Permission denied" when selecting photos
**Solution**:
1. Go to browser settings → Privacy & Security → Site Settings
2. Find this site and grant File System Access permission
3. Refresh page and try again

### Issue: Photos don't appear after selection
**Solution**:
1. Check browser console for errors
2. Verify photos are valid image formats (JPEG, PNG, HEIC)
3. Try with a different photo

### Issue: Drag-drop not working
**Solution**:
1. Ensure you have 2+ albums to reorder
2. Try clicking and holding for 300ms before dragging
3. Check if browser has disabled drag-drop (some privacy extensions)

### Issue: Album dates are wrong
**Solution**:
1. Check if photos have EXIF metadata (use exiftool or similar)
2. App falls back to file creation date if no EXIF
3. Manually check photo dates in file properties

### Issue: Performance is slow with large albums
**Solution**:
1. Clear browser cache and IndexedDB
2. Reduce thumbnail quality setting (if available)
3. Check if virtual scrolling is enabled (should be automatic for 100+ photos)

## Data Management

### Where is my data stored?
- **Album metadata**: Browser IndexedDB (local to your computer)
- **Photo files**: Remain in your file system (original location)
- **Thumbnails**: Cached in IndexedDB for performance

### How to backup data?
1. Export database: Click "Settings" → "Export Database"
2. Save the downloaded .sqlite file
3. To restore: "Settings" → "Import Database" → Select file

### How to clear all data?
1. Click "Settings" → "Clear All Data"
2. Confirm deletion
3. **Warning**: This removes all albums and metadata (photos on disk are safe)

## Next Steps

After completing this quickstart guide:
1. Review the data model in [data-model.md](./data-model.md)
2. Examine the database schema in [contracts/database-schema.sql](./contracts/database-schema.sql)
3. Read the implementation plan in [plan.md](./plan.md)
4. Begin development with `/speckit.tasks` to generate task breakdown
