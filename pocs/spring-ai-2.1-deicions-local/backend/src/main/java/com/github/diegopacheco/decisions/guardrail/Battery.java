package com.github.diegopacheco.decisions.guardrail;

import org.springaicommunity.decision.guardrail.DecisionGuardrail;

public enum Battery {
    INPUT, OUTPUT;

    DecisionGuardrail guardrail() {
        return switch (this) {
            case INPUT -> DecisionGuardrail.defaultInputBattery();
            case OUTPUT -> DecisionGuardrail.defaultOutputBattery();
        };
    }
}
