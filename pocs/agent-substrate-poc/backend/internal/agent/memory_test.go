package agent

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func fixedClock() func() time.Time {
	t := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	return func() time.Time {
		t = t.Add(time.Second)
		return t
	}
}

func TestJournalSurvivesAProcessRestartButMemoryDoesNot(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "data", "journal.log")
	first := NewMemory("boot-1", "host", journal, fixedClock())
	if _, err := first.Remember("buy milk"); err != nil {
		t.Fatal(err)
	}
	restarted := NewMemory("boot-2", "host", journal, fixedClock())
	state, err := restarted.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(state.MemoryNotes) != 0 {
		t.Errorf("a cold boot must start with empty memory, got %d notes", len(state.MemoryNotes))
	}
	if len(state.JournalNotes) != 1 || state.JournalNotes[0].Text != "buy milk" {
		t.Errorf("the durable journal must keep notes across boots, got %+v", state.JournalNotes)
	}
	if state.BootID != "boot-2" {
		t.Errorf("boot id identifies the process, got %s", state.BootID)
	}
}

func TestNoteIDsContinueFromTheJournal(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "journal.log")
	m := NewMemory("b", "h", journal, fixedClock())
	for _, text := range []string{"a", "b"} {
		if _, err := m.Remember(text); err != nil {
			t.Fatal(err)
		}
	}
	again := NewMemory("c", "h", journal, fixedClock())
	note, err := again.Remember("c")
	if err != nil {
		t.Fatal(err)
	}
	if note.ID != 3 {
		t.Errorf("ids must not restart after a cold boot, got %d", note.ID)
	}
}

func TestRememberRejectsBlankAndTruncatesLongNotes(t *testing.T) {
	m := NewMemory("b", "h", filepath.Join(t.TempDir(), "j.log"), fixedClock())
	tests := []struct {
		name    string
		in      string
		wantErr error
		wantLen int
	}{
		{"blank text is rejected", "   ", ErrEmptyNote, 0},
		{"long text is cut to the limit", strings.Repeat("x", 900), nil, maxNoteLength},
		{"surrounding spaces are trimmed", "  hi  ", nil, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			note, err := m.Remember(tt.in)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want %v", err, tt.wantErr)
			}
			if len(note.Text) != tt.wantLen {
				t.Errorf("len = %d, want %d", len(note.Text), tt.wantLen)
			}
		})
	}
}

func TestSnapshotCountsRequests(t *testing.T) {
	m := NewMemory("b", "h", filepath.Join(t.TempDir(), "j.log"), fixedClock())
	if _, err := m.Remember("x"); err != nil {
		t.Fatal(err)
	}
	state, err := m.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if state.Requests != 2 {
		t.Errorf("requests = %d, want 2", state.Requests)
	}
}
