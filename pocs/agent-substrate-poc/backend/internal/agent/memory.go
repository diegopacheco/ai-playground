package agent

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var ErrEmptyNote = errors.New("note text is empty")

const maxNoteLength = 500

type Note struct {
	ID   int       `json:"id"`
	Text string    `json:"text"`
	At   time.Time `json:"at"`
}

type State struct {
	BootID       string    `json:"bootId"`
	StartedAt    time.Time `json:"startedAt"`
	Hostname     string    `json:"hostname"`
	Requests     int       `json:"requests"`
	MemoryNotes  []Note    `json:"memoryNotes"`
	JournalNotes []Note    `json:"journalNotes"`
}

type Memory struct {
	mu        sync.Mutex
	bootID    string
	startedAt time.Time
	hostname  string
	journal   string
	now       func() time.Time
	requests  int
	notes     []Note
}

func NewMemory(bootID, hostname, journalPath string, now func() time.Time) *Memory {
	return &Memory{bootID: bootID, startedAt: now(), hostname: hostname, journal: journalPath, now: now}
}

func (m *Memory) Remember(text string) (Note, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return Note{}, ErrEmptyNote
	}
	if len(text) > maxNoteLength {
		text = text[:maxNoteLength]
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests++
	journal, err := m.readJournal()
	if err != nil {
		return Note{}, err
	}
	note := Note{ID: len(journal) + 1, Text: text, At: m.now()}
	if err := m.appendJournal(note); err != nil {
		return Note{}, err
	}
	m.notes = append(m.notes, note)
	return note, nil
}

func (m *Memory) Snapshot() (State, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests++
	journal, err := m.readJournal()
	if err != nil {
		return State{}, err
	}
	return State{
		BootID:       m.bootID,
		StartedAt:    m.startedAt,
		Hostname:     m.hostname,
		Requests:     m.requests,
		MemoryNotes:  append([]Note{}, m.notes...),
		JournalNotes: journal,
	}, nil
}

func (m *Memory) readJournal() ([]Note, error) {
	data, err := os.ReadFile(m.journal)
	if errors.Is(err, os.ErrNotExist) {
		return []Note{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read journal: %w", err)
	}
	notes := []Note{}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		var n Note
		if err := json.Unmarshal(scanner.Bytes(), &n); err != nil {
			return nil, fmt.Errorf("parse journal line %d: %w", len(notes)+1, err)
		}
		notes = append(notes, n)
	}
	return notes, scanner.Err()
}

func (m *Memory) appendJournal(n Note) error {
	if err := os.MkdirAll(filepath.Dir(m.journal), 0o755); err != nil {
		return fmt.Errorf("create journal dir: %w", err)
	}
	f, err := os.OpenFile(m.journal, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open journal: %w", err)
	}
	defer f.Close()
	line, err := json.Marshal(n)
	if err != nil {
		return fmt.Errorf("encode note: %w", err)
	}
	if _, err := f.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("append journal: %w", err)
	}
	return nil
}
