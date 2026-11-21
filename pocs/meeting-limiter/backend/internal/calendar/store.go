package calendar

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"sync"
)

type FileStore struct {
	mu       sync.Mutex
	path     string
	meetings []Meeting
}

func NewFileStore(path string) (*FileStore, error) {
	s := &FileStore{path: path}
	if path == "" {
		return s, nil
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if err := json.Unmarshal(data, &s.meetings); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return s, nil
}

func (s *FileStore) List(_ context.Context) ([]Meeting, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.meetings), nil
}

func (s *FileStore) Add(_ context.Context, m Meeting) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	next := append(slices.Clone(s.meetings), m)
	if err := s.flush(next); err != nil {
		return err
	}
	s.meetings = next
	return nil
}

func (s *FileStore) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := slices.IndexFunc(s.meetings, func(m Meeting) bool { return m.ID == id })
	if i < 0 {
		return fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	next := slices.Delete(slices.Clone(s.meetings), i, i+1)
	if err := s.flush(next); err != nil {
		return err
	}
	s.meetings = next
	return nil
}

func (s *FileStore) flush(meetings []Meeting) error {
	if s.path == "" {
		return nil
	}
	data, err := json.MarshalIndent(meetings, "", "  ")
	if err != nil {
		return fmt.Errorf("encode meetings: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("create data dir: %w", err)
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		return fmt.Errorf("replace %s: %w", s.path, err)
	}
	return nil
}
