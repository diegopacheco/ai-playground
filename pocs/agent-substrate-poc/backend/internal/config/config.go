package config

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Port           int
	AteBin         string
	KubeContext    string
	Atespace       string
	Template       string
	ProxyPortBase  int
	ProxyPortCount int
}

func Load(getenv func(string) string) (Config, error) {
	get := func(key, def string) string {
		if v := getenv(key); v != "" {
			return v
		}
		return def
	}
	port, err := strconv.Atoi(get("PORT", "8095"))
	if err != nil {
		return Config{}, fmt.Errorf("PORT: %w", err)
	}
	base, err := strconv.Atoi(get("PROXY_PORT_BASE", "18100"))
	if err != nil {
		return Config{}, fmt.Errorf("PROXY_PORT_BASE: %w", err)
	}
	count, err := strconv.Atoi(get("PROXY_PORT_COUNT", "50"))
	if err != nil {
		return Config{}, fmt.Errorf("PROXY_PORT_COUNT: %w", err)
	}
	return Config{
		Port:           port,
		AteBin:         get("ATE_BIN", "kubectl-ate"),
		KubeContext:    get("KUBE_CONTEXT", "kind-ate-poc"),
		Atespace:       get("ATESPACE", "ate-poc-agents"),
		Template:       get("TEMPLATE", "memory-agent"),
		ProxyPortBase:  base,
		ProxyPortCount: count,
	}, nil
}

func FromEnv() (Config, error) {
	return Load(os.Getenv)
}
