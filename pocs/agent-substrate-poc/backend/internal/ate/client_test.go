package ate

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
)

type fakeRunner struct {
	out   string
	err   error
	calls [][]string
}

func (f *fakeRunner) Run(_ context.Context, args ...string) ([]byte, error) {
	f.calls = append(f.calls, args)
	return []byte(f.out), f.err
}

const actorsJSON = `{"actors":[
 {"metadata":{"atespace":"poc","name":"alice","version":"7","createTime":"2026-11-27T08:00:00Z"},
  "actorTemplate":{"atespace":"poc","name":"memory-agent"},
  "status":{"state":"ACTOR_STATE_RUNNING","assignedNode":"node-1","snapshots":[{},{}],
   "workerAssignment":{"workerNamespace":"ate-poc-agents","workerPod":"memory-agent-abc"}}},
 {"metadata":{"atespace":"poc","name":"bob","version":3},
  "actorTemplate":{"atespace":"poc","name":"memory-agent"},
  "status":{"state":"ACTOR_STATE_SUSPENDED"}},
 {"metadata":{"atespace":"poc","name":"0b8c7a5e-1111-2222-3333-444455556666"},
  "actorTemplate":{"atespace":"poc","name":"memory-agent"},
  "status":{"state":"ACTOR_STATE_SUSPENDED"}},
 {"metadata":{"atespace":"poc","name":"other"},
  "actorTemplate":{"atespace":"poc","name":"counter"},
  "status":{"state":"ACTOR_STATE_RUNNING"}}
]}`

func TestListActorsParsesProtojsonAndHidesGoldenAndForeignActors(t *testing.T) {
	c := NewClient(&fakeRunner{out: actorsJSON}, "poc", "memory-agent")
	actors, err := c.ListActors(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(actors) != 2 {
		t.Fatalf("want alice and bob only, got %+v", actors)
	}
	alice := actors[0]
	if alice.State != StateRunning || alice.WorkerPod != "ate-poc-agents/memory-agent-abc" || alice.Version != 7 || alice.Snapshots != 2 {
		t.Errorf("alice parsed wrong: %+v", alice)
	}
	bob := actors[1]
	if bob.State != StateSuspended || bob.WorkerPod != "" || bob.Version != 3 {
		t.Errorf("a suspended actor has no worker: %+v", bob)
	}
}

func TestParseStateMapsUnknownValuesToUnknown(t *testing.T) {
	tests := []struct {
		in   string
		want State
	}{
		{"ACTOR_STATE_RESUMING", StateResuming},
		{"ACTOR_STATE_CRASHED", StateCrashed},
		{"ACTOR_STATE_SOMETHING_NEW", StateUnknown},
		{"", StateUnknown},
	}
	for _, tt := range tests {
		if got := parseState(tt.in); got != tt.want {
			t.Errorf("parseState(%q) = %s, want %s", tt.in, got, tt.want)
		}
	}
}

func TestCommandsTargetTheConfiguredAtespaceAndTemplate(t *testing.T) {
	r := &fakeRunner{}
	c := NewClient(r, "poc", "memory-agent")
	ctx := context.Background()
	if err := c.CreateActor(ctx, "alice"); err != nil {
		t.Fatal(err)
	}
	if err := c.SuspendActor(ctx, "alice"); err != nil {
		t.Fatal(err)
	}
	want := [][]string{
		{"create", "actor", "alice", "--template", "memory-agent", "-a", "poc"},
		{"suspend", "actor", "alice", "-a", "poc"},
	}
	if !reflect.DeepEqual(r.calls, want) {
		t.Errorf("calls = %v, want %v", r.calls, want)
	}
}

func TestCLIErrorsAreClassifiedFromGrpcCodes(t *testing.T) {
	tests := []struct {
		output string
		want   error
	}{
		{"rpc error: code = NotFound desc = actor alice", ErrNotFound},
		{"rpc error: code = AlreadyExists desc = actor alice", ErrAlreadyExists},
		{"rpc error: code = FailedPrecondition desc = actor is not running", ErrConflict},
	}
	for _, tt := range tests {
		err := &CommandError{Args: []string{"x"}, Output: tt.output, kind: classify(tt.output)}
		c := NewClient(&fakeRunner{err: err}, "poc", "t")
		got := c.SuspendActor(context.Background(), "alice")
		if !errors.Is(got, tt.want) {
			t.Errorf("%q: got %v, want %v", tt.output, got, tt.want)
		}
		if !strings.Contains(got.Error(), "suspend actor alice") {
			t.Errorf("error must say what failed: %v", got)
		}
	}
}

func TestListWorkersParsesCapacity(t *testing.T) {
	out := `{"workers":[{"metadata":{"name":"w1"},"workerPool":"memory-agent","workerPod":"p1","nodeName":"n1",
	  "status":{"state":"WORKER_STATE_ACTIVE","allocated":{"actors":1},"capacity":{"actors":"1"}}}]}`
	workers, err := NewClient(&fakeRunner{out: out}, "poc", "t").ListWorkers(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := Worker{Name: "w1", Pool: "memory-agent", Pod: "p1", Node: "n1", State: "ACTIVE", Actors: 1, Capacity: 1}
	if workers[0] != want {
		t.Errorf("got %+v, want %+v", workers[0], want)
	}
}
