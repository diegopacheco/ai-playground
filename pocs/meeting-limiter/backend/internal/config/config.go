package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Port     int
	DataFile string
}

func Load() (Config, error) {
	port, err := strconv.Atoi(os.Getenv("PORT"))
	if err != nil {
		return Config{}, fmt.Errorf("PORT: %w", err)
	}
	data := os.Getenv("DATA_FILE")
	if data == "" {
		return Config{}, errors.New("DATA_FILE is required")
	}
	return Config{Port: port, DataFile: data}, nil
}
