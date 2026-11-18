package com.github.diegopacheco.restate.state;

import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Locale;

public enum StepStatus {
    PENDING, RUNNING, DONE;

    @JsonValue
    public String json() {
        return name().toLowerCase(Locale.ROOT);
    }
}
