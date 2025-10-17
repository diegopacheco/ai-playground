# Tasks: Photo Album Organizer

**Input**: Design documents from `/specs/001-photo-album-organizer/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/database-schema.sql, research.md, quickstart.md

**Tests**: Per constitution (TDD NON-NEGOTIABLE), tests are included before implementation tasks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions
- Single web project: `src/`, `tests/` at repository root (photoalbum/)
- Tests use Vitest (unit) and Playwright (integration)

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Initialize Vite project with JavaScript in photoalbum/
- [X] T002 [P] Create directory structure in photoalbum/src/ (components/, services/, db/, utils/, styles/)
- [X] T003 [P] Create test directory structure in photoalbum/tests/ (unit/, integration/, fixtures/)
- [X] T004 [P] Install dependencies: sql.js, exifr in photoalbum/package.json
- [X] T005 [P] Install dev dependencies: vitest, playwright, @playwright/test in photoalbum/package.json
- [X] T006 Configure Vitest in photoalbum/vite.config.js
- [X] T007 [P] Configure Playwright in photoalbum/playwright.config.js
- [X] T008 [P] Create base HTML template in photoalbum/src/index.html
- [X] T009 [P] Create main application entry in photoalbum/src/main.js
- [X] T010 [P] Create base CSS structure in photoalbum/src/styles/main.css

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T011 Initialize sql.js database module in photoalbum/src/db/init.js
- [X] T012 Implement database schema loader in photoalbum/src/db/schema.js
- [X] T013 Create IndexedDB persistence layer in photoalbum/src/db/persistence.js
- [X] T014 [P] Implement EXIF metadata extractor in photoalbum/src/utils/exif-reader.js
- [X] T015 [P] Implement thumbnail generator using Canvas API in photoalbum/src/utils/thumbnail-generator.js
- [X] T016 [P] Implement File System Access API wrapper in photoalbum/src/utils/file-handler.js
- [X] T017 Create database access layer base in photoalbum/src/db/database.js
- [X] T018 [P] Write unit test for database initialization in photoalbum/tests/unit/db-init.test.js
- [X] T019 [P] Write unit test for EXIF reader in photoalbum/tests/unit/exif-reader.test.js
- [X] T020 [P] Write unit test for thumbnail generator in photoalbum/tests/unit/thumbnail-generator.test.js

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Create and View Photo Albums (Priority: P1) 🎯 MVP

**Goal**: Enable users to create albums, add photos, and view them in a tile grid

**Independent Test**: Create an album, add photos, view photos in tile grid - complete workflow functional

### Tests for User Story 1 (TDD - Write FIRST)

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T021 [P] [US1] Contract test for Album table schema in photoalbum/tests/unit/album-model.test.js
- [ ] T022 [P] [US1] Contract test for Photo table schema in photoalbum/tests/unit/photo-model.test.js
- [ ] T023 [P] [US1] Contract test for AlbumPhoto junction table in photoalbum/tests/unit/album-photo-model.test.js
- [ ] T024 [P] [US1] Unit test for AlbumService.createAlbum() in photoalbum/tests/unit/album-service.test.js
- [ ] T025 [P] [US1] Unit test for PhotoService.addPhoto() in photoalbum/tests/unit/photo-service.test.js
- [ ] T026 [P] [US1] Unit test for AlbumService.addPhotoToAlbum() in photoalbum/tests/unit/album-service.test.js
- [ ] T027 [P] [US1] Unit test for AlbumService.getAlbumPhotos() in photoalbum/tests/unit/album-service.test.js
- [ ] T028 [US1] Integration test for create album workflow in photoalbum/tests/integration/create-album.spec.js
- [ ] T029 [US1] Integration test for add photos to album workflow in photoalbum/tests/integration/add-photos.spec.js
- [ ] T030 [US1] Integration test for view album photos in tile grid in photoalbum/tests/integration/view-album.spec.js

### Implementation for User Story 1

- [ ] T031 [P] [US1] Implement Album database operations in photoalbum/src/db/album-db.js
- [ ] T032 [P] [US1] Implement Photo database operations in photoalbum/src/db/photo-db.js
- [ ] T033 [US1] Implement AlbumService with createAlbum method in photoalbum/src/services/album-service.js
- [ ] T034 [US1] Implement PhotoService with addPhoto method in photoalbum/src/services/photo-service.js
- [ ] T035 [US1] Implement AlbumService.addPhotoToAlbum method in photoalbum/src/services/album-service.js
- [ ] T036 [US1] Implement AlbumService.getAlbumPhotos method in photoalbum/src/services/album-service.js
- [ ] T037 [P] [US1] Create AlbumGrid component for main page in photoalbum/src/components/album-grid.js
- [ ] T038 [P] [US1] Create AlbumCard component for album display in photoalbum/src/components/album-card.js
- [ ] T039 [US1] Create PhotoTileGrid component for photo display in photoalbum/src/components/photo-tile-grid.js
- [ ] T040 [US1] Create PhotoTile component for individual photos in photoalbum/src/components/photo-tile.js
- [ ] T041 [US1] Implement CreateAlbumModal component in photoalbum/src/components/create-album-modal.js
- [ ] T042 [US1] Implement AddPhotosButton component with File System Access in photoalbum/src/components/add-photos-button.js
- [ ] T043 [US1] Wire up album creation in main.js event handlers in photoalbum/src/main.js
- [ ] T044 [US1] Wire up photo addition in album view in photoalbum/src/main.js
- [ ] T045 [US1] Add CSS styling for album grid in photoalbum/src/styles/album-grid.css
- [ ] T046 [US1] Add CSS styling for photo tile grid in photoalbum/src/styles/photo-grid.css
- [ ] T047 [US1] Implement virtual scrolling for large photo grids in photoalbum/src/components/virtual-scroller.js
- [ ] T048 [US1] Add error handling and user feedback for file access in photoalbum/src/utils/error-handler.js

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Reorganize Albums via Drag and Drop (Priority: P2)

**Goal**: Enable drag-and-drop reordering of albums on main page with persistence

**Independent Test**: Create multiple albums, drag to reorder, verify persistence across page refresh

### Tests for User Story 2 (TDD - Write FIRST)

- [ ] T049 [P] [US2] Unit test for AlbumService.updateDisplayOrder() in photoalbum/tests/unit/album-service.test.js
- [ ] T050 [P] [US2] Unit test for drag-drop state management in photoalbum/tests/unit/drag-handler.test.js
- [ ] T051 [US2] Integration test for album drag-drop reorder in photoalbum/tests/integration/reorder-albums.spec.js
- [ ] T052 [US2] Integration test for drag-drop persistence in photoalbum/tests/integration/persist-order.spec.js

### Implementation for User Story 2

- [ ] T053 [US2] Implement AlbumService.updateDisplayOrder method in photoalbum/src/services/album-service.js
- [ ] T054 [US2] Implement AlbumService.reorderAlbums batch update in photoalbum/src/services/album-service.js
- [ ] T055 [US2] Create DragDropHandler utility in photoalbum/src/utils/drag-drop-handler.js
- [ ] T056 [US2] Add draggable attributes to AlbumCard component in photoalbum/src/components/album-card.js
- [ ] T057 [US2] Implement drag event handlers (dragstart, dragover, drop) in photoalbum/src/components/album-grid.js
- [ ] T058 [US2] Add visual feedback during drag (CSS classes) in photoalbum/src/styles/drag-drop.css
- [ ] T059 [US2] Implement drop zone indicator in photoalbum/src/components/album-grid.js
- [ ] T060 [US2] Add keyboard navigation fallback for reordering in photoalbum/src/utils/keyboard-reorder.js
- [ ] T061 [US2] Debounce database updates during rapid reordering in photoalbum/src/utils/debounce.js
- [ ] T062 [US2] Add ARIA labels for accessibility in photoalbum/src/components/album-card.js

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Automatic Date-Based Album Grouping (Priority: P3)

**Goal**: Group albums by month/year based on photo dates with toggle functionality

**Independent Test**: Create albums with photos from different dates, enable date grouping, verify chronological organization

### Tests for User Story 3 (TDD - Write FIRST)

- [ ] T063 [P] [US3] Unit test for date group computation in photoalbum/tests/unit/date-grouping.test.js
- [ ] T064 [P] [US3] Unit test for AlbumService.getAlbumsGroupedByDate() in photoalbum/tests/unit/album-service.test.js
- [ ] T065 [US3] Integration test for date grouping display in photoalbum/tests/integration/date-grouping.spec.js
- [ ] T066 [US3] Integration test for drag-drop within date groups in photoalbum/tests/integration/reorder-in-groups.spec.js

### Implementation for User Story 3

- [ ] T067 [US3] Implement AlbumService.getAlbumsGroupedByDate query in photoalbum/src/services/album-service.js
- [ ] T068 [US3] Implement date formatting utility with Intl.DateTimeFormat in photoalbum/src/utils/date-formatter.js
- [ ] T069 [US3] Create DateGroupHeader component in photoalbum/src/components/date-group-header.js
- [ ] T070 [US3] Update AlbumGrid to support grouped display in photoalbum/src/components/album-grid.js
- [ ] T071 [US3] Create GroupingToggle component in photoalbum/src/components/grouping-toggle.js
- [ ] T072 [US3] Add state management for grouping toggle in photoalbum/src/main.js
- [ ] T073 [US3] Restrict drag-drop to within date groups in photoalbum/src/utils/drag-drop-handler.js
- [ ] T074 [US3] Add CSS styling for date group headers in photoalbum/src/styles/date-groups.css
- [ ] T075 [US3] Handle albums with unknown dates (fallback group) in photoalbum/src/services/album-service.js

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T076 [P] Add loading states for async operations in photoalbum/src/components/loading-spinner.js
- [ ] T077 [P] Implement error boundary and user-friendly error messages in photoalbum/src/utils/error-handler.js
- [ ] T078 [P] Add empty state placeholders for no albums/photos in photoalbum/src/components/empty-state.js
- [ ] T079 [P] Implement delete album functionality in photoalbum/src/services/album-service.js
- [ ] T080 [P] Implement remove photo from album in photoalbum/src/services/album-service.js
- [ ] T081 Add confirmation dialogs for destructive actions in photoalbum/src/components/confirm-dialog.js
- [ ] T082 [P] Optimize thumbnail caching strategy in photoalbum/src/utils/thumbnail-cache.js
- [ ] T083 [P] Add performance monitoring with Performance API in photoalbum/src/utils/performance-monitor.js
- [ ] T084 [P] Implement accessibility features (ARIA labels, focus management) across components
- [ ] T085 [P] Add responsive CSS for different screen sizes in photoalbum/src/styles/responsive.css
- [ ] T086 Run Lighthouse CI for bundle size validation (<250KB target)
- [ ] T087 Run axe-core accessibility audit and fix violations
- [ ] T088 Add test fixtures with sample images in photoalbum/tests/fixtures/
- [ ] T089 [P] Write unit tests for edge cases (duplicate photos, missing metadata) in photoalbum/tests/unit/edge-cases.test.js
- [ ] T090 Run full integration test suite and validate quickstart.md scenarios

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3, 4, 5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Integrates with US1 (AlbumGrid component)
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Integrates with US1 and US2 (AlbumGrid, drag-drop)

### Within Each User Story

- Tests (T021-T030, T049-T052, T063-T066) MUST be written and FAIL before implementation
- Database operations before services
- Services before components
- Components before UI wiring
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Database operations [P] can run in parallel (different files)
- Components marked [P] can run in parallel (different files)
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all unit tests for User Story 1 together:
Task T021: "Contract test for Album table schema"
Task T022: "Contract test for Photo table schema"
Task T023: "Contract test for AlbumPhoto junction table"
Task T024: "Unit test for AlbumService.createAlbum()"
Task T025: "Unit test for PhotoService.addPhoto()"

# Launch database operations together (after tests fail):
Task T031: "Implement Album database operations"
Task T032: "Implement Photo database operations"

# Launch components together (after services complete):
Task T037: "Create AlbumGrid component"
Task T038: "Create AlbumCard component"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently using quickstart.md
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2 (can start, will integrate later)
   - Developer C: User Story 3 (can start, will integrate later)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- **TDD MANDATORY**: Verify tests fail before implementing (Constitution Principle II)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
