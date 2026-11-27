package ate

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"
)

type State string

const (
	StateResuming   State = "RESUMING"
	StateRunning    State = "RUNNING"
	StateSuspending State = "SUSPENDING"
	StateSuspended  State = "SUSPENDED"
	StateCrashed    State = "CRASHED"
	StateDeleting   State = "DELETING"
	StateReverting  State = "REVERTING"
	StateUnknown    State = "UNKNOWN"
)

var knownStates = []State{StateResuming, StateRunning, StateSuspending, StateSuspended, StateCrashed, StateDeleting, StateReverting}

func parseState(raw string) State {
	s := State(strings.TrimPrefix(raw, "ACTOR_STATE_"))
	for _, k := range knownStates {
		if s == k {
			return k
		}
	}
	return StateUnknown
}

type Actor struct {
	Name      string    `json:"name"`
	Template  string    `json:"template"`
	State     State     `json:"state"`
	WorkerPod string    `json:"workerPod"`
	Node      string    `json:"node"`
	Version   int64     `json:"version"`
	Snapshots int       `json:"snapshots"`
	CreatedAt time.Time `json:"createdAt"`
}

type Worker struct {
	Name     string `json:"name"`
	Pool     string `json:"pool"`
	Pod      string `json:"pod"`
	Node     string `json:"node"`
	State    string `json:"state"`
	Actors   int64  `json:"actors"`
	Capacity int64  `json:"capacity"`
}

type flexInt int64

func (f *flexInt) UnmarshalJSON(b []byte) error {
	s := strings.Trim(string(b), "\"")
	if s == "" || s == "null" {
		*f = 0
		return nil
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return err
	}
	*f = flexInt(n)
	return nil
}

type wireMetadata struct {
	Name       string    `json:"name"`
	Atespace   string    `json:"atespace"`
	Version    flexInt   `json:"version"`
	CreateTime time.Time `json:"createTime"`
}

type wireActor struct {
	Metadata      wireMetadata `json:"metadata"`
	ActorTemplate struct {
		Atespace string `json:"atespace"`
		Name     string `json:"name"`
	} `json:"actorTemplate"`
	Status struct {
		State            string            `json:"state"`
		AssignedNode     string            `json:"assignedNode"`
		Snapshots        []json.RawMessage `json:"snapshots"`
		WorkerAssignment *struct {
			WorkerNamespace string `json:"workerNamespace"`
			WorkerPod       string `json:"workerPod"`
		} `json:"workerAssignment"`
	} `json:"status"`
}

type wireActors struct {
	Actors []wireActor `json:"actors"`
}

type wireWorker struct {
	Metadata   wireMetadata `json:"metadata"`
	WorkerPool string       `json:"workerPool"`
	WorkerPod  string       `json:"workerPod"`
	NodeName   string       `json:"nodeName"`
	Status     struct {
		State     string `json:"state"`
		Allocated struct {
			Actors flexInt `json:"actors"`
		} `json:"allocated"`
		Capacity struct {
			Actors flexInt `json:"actors"`
		} `json:"capacity"`
	} `json:"status"`
}

type wireWorkers struct {
	Workers []wireWorker `json:"workers"`
}

func (w wireActor) toActor() Actor {
	a := Actor{
		Name:      w.Metadata.Name,
		Template:  w.ActorTemplate.Name,
		State:     parseState(w.Status.State),
		Node:      w.Status.AssignedNode,
		Version:   int64(w.Metadata.Version),
		Snapshots: len(w.Status.Snapshots),
		CreatedAt: w.Metadata.CreateTime,
	}
	if w.Status.WorkerAssignment != nil && w.Status.WorkerAssignment.WorkerPod != "" {
		a.WorkerPod = w.Status.WorkerAssignment.WorkerNamespace + "/" + w.Status.WorkerAssignment.WorkerPod
	}
	return a
}

func (w wireWorker) toWorker() Worker {
	return Worker{
		Name:     w.Metadata.Name,
		Pool:     w.WorkerPool,
		Pod:      w.WorkerPod,
		Node:     w.NodeName,
		State:    strings.TrimPrefix(w.Status.State, "WORKER_STATE_"),
		Actors:   int64(w.Status.Allocated.Actors),
		Capacity: int64(w.Status.Capacity.Actors),
	}
}
