package com.game.terminator.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.Transient;
import org.springframework.data.domain.Persistable;
import org.springframework.data.relational.core.mapping.Table;

@Table("games")
public class Game implements Persistable<String> {
    @Id
    private String id;
    private String terminatorAgent;
    private String terminatorModel;
    private String mosquitoAgent;
    private String mosquitoModel;
    private int gridSize;
    private String winner;
    private int totalCycles;
    private int maxMosquitos;
    private int totalKills;
    private int totalHatched;
    private int totalDates;
    private String status;
    private String createdAt;
    private String endedAt;

    @Transient
    private boolean isNew = false;

    public Game() {}

    public Game(String id, String terminatorAgent, String terminatorModel,
                String mosquitoAgent, String mosquitoModel, int gridSize,
                String status, String createdAt) {
        this.id = id;
        this.terminatorAgent = terminatorAgent;
        this.terminatorModel = terminatorModel;
        this.mosquitoAgent = mosquitoAgent;
        this.mosquitoModel = mosquitoModel;
        this.gridSize = gridSize;
        this.status = status;
        this.createdAt = createdAt;
        this.isNew = true;
    }

    @Override
    public String getId() { return id; }

    @Override
    public boolean isNew() { return isNew; }

    public void setId(String id) { this.id = id; }
    public String getTerminatorAgent() { return terminatorAgent; }
    public void setTerminatorAgent(String terminatorAgent) { this.terminatorAgent = terminatorAgent; }
    public String getTerminatorModel() { return terminatorModel; }
    public void setTerminatorModel(String terminatorModel) { this.terminatorModel = terminatorModel; }
    public String getMosquitoAgent() { return mosquitoAgent; }
    public void setMosquitoAgent(String mosquitoAgent) { this.mosquitoAgent = mosquitoAgent; }
    public String getMosquitoModel() { return mosquitoModel; }
    public void setMosquitoModel(String mosquitoModel) { this.mosquitoModel = mosquitoModel; }
    public int getGridSize() { return gridSize; }
    public void setGridSize(int gridSize) { this.gridSize = gridSize; }
    public String getWinner() { return winner; }
    public void setWinner(String winner) { this.winner = winner; }
    public int getTotalCycles() { return totalCycles; }
    public void setTotalCycles(int totalCycles) { this.totalCycles = totalCycles; }
    public int getMaxMosquitos() { return maxMosquitos; }
    public void setMaxMosquitos(int maxMosquitos) { this.maxMosquitos = maxMosquitos; }
    public int getTotalKills() { return totalKills; }
    public void setTotalKills(int totalKills) { this.totalKills = totalKills; }
    public int getTotalHatched() { return totalHatched; }
    public void setTotalHatched(int totalHatched) { this.totalHatched = totalHatched; }
    public int getTotalDates() { return totalDates; }
    public void setTotalDates(int totalDates) { this.totalDates = totalDates; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getCreatedAt() { return createdAt; }
    public void setCreatedAt(String createdAt) { this.createdAt = createdAt; }
    public String getEndedAt() { return endedAt; }
    public void setEndedAt(String endedAt) { this.endedAt = endedAt; }
    public void markNotNew() { this.isNew = false; }
}
