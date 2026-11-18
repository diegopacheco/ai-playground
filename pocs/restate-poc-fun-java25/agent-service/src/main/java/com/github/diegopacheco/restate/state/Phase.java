package com.github.diegopacheco.restate.state;

import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Locale;

public enum Phase {
    PLANNING, WORKING, SUMMARIZING, DONE, FAILED;

    @JsonValue
    public String json() {
        return name().toLowerCase(Locale.ROOT);
    }
}
