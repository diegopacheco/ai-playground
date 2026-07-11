package com.diegopacheco.temporalpoc.workflow;

import io.temporal.activity.ActivityInterface;

@ActivityInterface
public interface NewsAgentActivity {
    String researchNews(String symbol, String company);
}
