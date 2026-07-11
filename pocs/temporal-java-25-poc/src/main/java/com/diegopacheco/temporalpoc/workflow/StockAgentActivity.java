package com.diegopacheco.temporalpoc.workflow;

import io.temporal.activity.ActivityInterface;

@ActivityInterface
public interface StockAgentActivity {
    String researchStock(String symbol, String company);
}
