package com.github.diegopacheco.decisions.triage;

public enum Route {
    AUTO_REPAIR_TICKET, HUMAN_REVIEW, NO_ACTION;

    public static Route of(double defectProbability, double autoThreshold, double reviewThreshold) {
        if (defectProbability >= autoThreshold) {
            return AUTO_REPAIR_TICKET;
        }
        if (defectProbability >= reviewThreshold) {
            return HUMAN_REVIEW;
        }
        return NO_ACTION;
    }
}
