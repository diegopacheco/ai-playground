//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"testing"
	"time"

	"agentsubstratepoc/internal/agent"
	"agentsubstratepoc/internal/ate"
)

var client = &http.Client{Timeout: 3 * time.Minute}

func call(t *testing.T, method, path string, body any, want int, out any) {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	req, err := http.NewRequest(method, os.Getenv("CONSOLE_URL")+path, &buf)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != want {
		var e map[string]string
		_ = json.NewDecoder(resp.Body).Decode(&e)
		t.Fatalf("%s %s: status %d want %d: %v", method, path, resp.StatusCode, want, e)
	}
	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			t.Fatal(err)
		}
	}
}

func waitState(t *testing.T, name string, want ate.State) ate.Actor {
	t.Helper()
	deadline := time.Now().Add(3 * time.Minute)
	for {
		var a ate.Actor
		call(t, "GET", "/api/actors/"+name, nil, http.StatusOK, &a)
		if a.State == want {
			return a
		}
		if time.Now().After(deadline) {
			t.Fatalf("actor %s stuck in %s, want %s", name, a.State, want)
		}
		time.Sleep(time.Second)
	}
}

func TestMemorySurvivesSuspendAndResume(t *testing.T) {
	name := fmt.Sprintf("e2e-%d", time.Now().Unix())
	t.Cleanup(func() { call(t, "DELETE", "/api/actors/"+name, nil, http.StatusOK, nil) })
	call(t, "POST", "/api/actors", map[string]string{"name": name}, http.StatusCreated, nil)

	call(t, "POST", "/api/actors/"+name+"/notes", map[string]string{"text": "first"}, http.StatusCreated, nil)
	var before agent.State
	call(t, "GET", "/api/actors/"+name+"/state", nil, http.StatusOK, &before)
	if len(before.MemoryNotes) != 1 || len(before.JournalNotes) != 1 {
		t.Fatalf("after one note: %+v", before)
	}

	call(t, "POST", "/api/actors/"+name+"/suspend", nil, http.StatusOK, nil)
	suspended := waitState(t, name, ate.StateSuspended)
	if suspended.WorkerPod != "" {
		t.Errorf("a suspended actor must free its worker, still on %s", suspended.WorkerPod)
	}

	call(t, "POST", "/api/actors/"+name+"/notes", map[string]string{"text": "second"}, http.StatusCreated, nil)
	running := waitState(t, name, ate.StateRunning)
	if running.Version <= suspended.Version {
		t.Errorf("resume must bump the version: %d -> %d", suspended.Version, running.Version)
	}
	var after agent.State
	call(t, "GET", "/api/actors/"+name+"/state", nil, http.StatusOK, &after)
	if after.BootID != before.BootID {
		t.Errorf("memory fidelity restores the same process, boot id %s -> %s", before.BootID, after.BootID)
	}
	if len(after.MemoryNotes) != 2 || after.MemoryNotes[0].Text != "first" {
		t.Errorf("RAM notes must survive hibernation: %+v", after.MemoryNotes)
	}
	if len(after.JournalNotes) != 2 {
		t.Errorf("journal must have both notes: %+v", after.JournalNotes)
	}
}

func TestInvalidAndUnknownActors(t *testing.T) {
	call(t, "POST", "/api/actors", map[string]string{"name": "Bad_Name"}, http.StatusBadRequest, nil)
	call(t, "GET", "/api/actors/does-not-exist-42", nil, http.StatusNotFound, nil)
}
