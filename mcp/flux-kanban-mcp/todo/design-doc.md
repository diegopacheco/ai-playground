# Grocery Todo List App - Design Document

## Overview

A grocery todo list application built with modern TypeScript tooling. All features listed in this document must become tasks in the Flux Kanban project called **todo**. If the project does not exist, create it first. Each feature section maps directly to a Flux task. The LLM implementing this must create all tasks in Flux, implement each task, and update task status in Flux as work progresses.

---

## Tech Stack

- **Runtime**: Bun
- **Bundler/Dev Server**: Vite
- **UI Framework**: React
- **State/Data Management**: TanStack (TanStack Query, TanStack Router, TanStack Table as needed)
- **Language**: TypeScript throughout — no JavaScript files

---

## Features

### 1. Add New Items

The app must allow users to receive and add new grocery items to a list. Items have at minimum a name and a completion status.

**Flux Task**: "Add new items to a grocery list"

---

### 2. Delete Items or Mark as Done

Users must be able to delete items from a list or mark them as completed/done. Completed items should be visually distinct from pending items.

**Flux Task**: "Delete items or mark them as done"

---

### 3. Multiple Lists

The app must support managing multiple independent grocery lists. Users can create, rename, and delete lists. Each list holds its own set of items.

**Flux Task**: "Support multiple grocery lists"

---

### 4. Search

Users must be able to search across items within a list (or across all lists). Results should update in real time as the user types.

**Flux Task**: "Search items across lists"

---

### 5. Export Lists to PDF

Users must be able to export any grocery list to a PDF file. The PDF should be clean and readable.

**Flux Task**: "Export lists to PDF"

---

### 6. Image Drop — AI Item Extraction

Users must be able to drop an image onto the app. The app will send the image to a backend TypeScript service that uses an AI/LLM to extract grocery items from the image and add them to the current list automatically.

Reference implementation pattern (in Rust, to be rewritten fully in TypeScript):
https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-debate-club

The entire solution — frontend and backend AI service — must be TypeScript. No Rust, no Python.

**Flux Task**: "Image drop — AI extracts items from image into list"

---

### 7. run.sh and stop.sh

The project must include:
- `run.sh`: starts the full application (frontend + any backend services)
- `stop.sh`: stops all running processes cleanly

**Flux Task**: "Add run.sh and stop.sh scripts"

---

## Flux Task Instructions for the Implementing LLM

The implementing LLM **must** follow this process:

1. **Check if the Flux project `todo` exists.** If it does not, create it.
2. **Create one Flux task per feature** listed above using the exact task titles specified in each feature section.
3. **Before starting implementation of any feature**, move the corresponding Flux task to `in_progress`.
4. **After completing implementation of a feature**, move the corresponding Flux task to `done`.
5. **Never mark a task as done unless it is fully implemented and working.**
6. Tasks must be created in this order (dependencies first):
   - Tech stack setup (Vite + Bun + React + TanStack)
   - Add new items
   - Delete items or mark as done
   - Multiple lists
   - Search
   - Export to PDF
   - run.sh and stop.sh
   - Image drop AI extraction (last, most complex)

All implementation work lives inside the `todo/` directory relative to this design document.
