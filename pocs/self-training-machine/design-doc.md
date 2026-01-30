# Self Training Machine - Design Document

## Overview

Self Training Machine is an AI-powered training and quiz generation system. Users provide a topic and the system generates comprehensive training content with an assessment quiz.

## Features

### 1. Prompt UI
- Text area for custom training topic input
- Word and character count display
- 4 suggested training buttons:
  - Using AI to write better emails
  - Using AI with Excel spreadsheets
  - Using AI to improve your work productivity
  - Using AI to review PPT presentations

### 2. Training UI
- Split layout with topics navigation on the left
- Training content display on the right
- Bottom panel with Q&A functionality for asking questions about the content
- "Take Quiz" button to proceed to assessment

### 3. Quiz UI
- 10 multiple choice questions generated from training content
- Navigation between questions
- Progress indicator
- Score calculation at submission
- Pass threshold: 70%

### 4. Certificate UI
- Professional certificate of completion
- Displays: user name, training title, score, percentage, date
- Unique certificate ID
- Print functionality

## Architecture

### Backend
- **Language**: Rust Edition 2024 (v1.93)
- **Framework**: Axum 0.8
- **Async Runtime**: Tokio
- **Real-time**: Server-Sent Events (SSE)
- **AI Integration**: Claude CLI with Opus 4.5 model

### Frontend
- **Framework**: React 19
- **Styling**: TailwindCSS 3.4
- **Routing**: TanStack Router
- **Language**: TypeScript 5.7
- **Build Tool**: Vite 6
- **Package Manager**: Bun

## API Endpoints

### POST /api/generate
SSE endpoint that generates training content and quiz.

Request:
```json
{
  "prompt": "Training topic"
}
```

SSE Events:
- `start` - Initial event with total steps
- `progress` - Step progress updates
- `training_ready` - Training content payload
- `quiz_ready` - Quiz questions payload
- `error` - Error messages

### POST /api/ask
Answer questions about training content.

Request:
```json
{
  "question": "User question",
  "context": "Training context"
}
```

Response:
```json
{
  "answer": "AI generated answer"
}
```

### POST /api/submit-quiz
Submit quiz answers for grading.

Request:
```json
{
  "answers": [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]
}
```

Response:
```json
{
  "score": 8,
  "total": 10,
  "percentage": 80.0,
  "passed": true
}
```

### POST /api/certificate
Generate completion certificate.

Request:
```json
{
  "user_name": "John Doe",
  "training_title": "Using AI to Write Better Emails",
  "score": 10,
  "total": 10,
  "percentage": 100.0
}
```

Response:
```json
{
  "id": "uuid",
  "user_name": "John Doe",
  "training_title": "Using AI to Write Better Emails",
  "score": 10,
  "total": 10,
  "percentage": 100.0,
  "date": "2026-01-30"
}
```

### GET /api/health
Health check endpoint.

## Data Flow

```
User Input (Prompt)
    ↓
Frontend POST /api/generate (SSE request)
    ↓
Backend calls Claude CLI for training generation
    ↓
Send training_ready event
    ↓
Backend calls Claude CLI for quiz generation
    ↓
Send quiz_ready event
    ↓
User reads training content
    ↓
User takes quiz
    ↓
POST /api/submit-quiz
    ↓
Calculate score
    ↓
If passed: POST /api/certificate
    ↓
Display certificate
```

## AI Integration

The system uses the Claude CLI tool with the claude-opus-4-5-20251101 model.

Training Generation Prompt:
- Creates 5 topics with detailed educational content
- Each topic includes practical information and tips
- Returns structured JSON

Quiz Generation Prompt:
- Creates 10 multiple choice questions
- Each question has 4 options
- Correct answer index included
- Returns structured JSON

Q&A Prompt:
- Uses training content as context
- Answers questions about the material

## UI Components

- **PromptView**: Initial input screen
- **TrainingView**: Training content display with Q&A
- **QuizView**: Interactive quiz with navigation
- **CertificateView**: Printable certificate
- **LoadingView**: Progress indicator during generation

## Running the Application

```bash
./run.sh
```

Backend: http://localhost:8080
Frontend: http://localhost:3000

## Requirements

- Rust 1.85+
- Bun
- Claude CLI configured with API access
