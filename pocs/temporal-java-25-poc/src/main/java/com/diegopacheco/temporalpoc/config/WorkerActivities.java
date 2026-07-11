package com.diegopacheco.temporalpoc.config;

import com.diegopacheco.temporalpoc.workflow.DecisionAgentActivityImpl;
import com.diegopacheco.temporalpoc.workflow.NewsAgentActivityImpl;
import com.diegopacheco.temporalpoc.workflow.StockAgentActivityImpl;
import org.springframework.stereotype.Component;

@Component
public class WorkerActivities {
    private final StockAgentActivityImpl stock;
    private final NewsAgentActivityImpl news;
    private final DecisionAgentActivityImpl decision;

    public WorkerActivities(StockAgentActivityImpl stock, NewsAgentActivityImpl news, DecisionAgentActivityImpl decision) {
        this.stock = stock;
        this.news = news;
        this.decision = decision;
    }

    public StockAgentActivityImpl stock() {
        return stock;
    }

    public NewsAgentActivityImpl news() {
        return news;
    }

    public DecisionAgentActivityImpl decision() {
        return decision;
    }
}
