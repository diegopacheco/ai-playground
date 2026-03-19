package com.game.terminator.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.relational.core.mapping.Table;

@Table("game_events")
public class GameEvent {
    @Id
    private Long id;
    private String gameId;
    private int cycle;
    private String eventType;
    private String eventData;
    private String createdAt;

    public GameEvent() {}

    public GameEvent(String gameId, int cycle, String eventType, String eventData, String createdAt) {
        this.gameId = gameId;
        this.cycle = cycle;
        this.eventType = eventType;
        this.eventData = eventData;
        this.createdAt = createdAt;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getGameId() { return gameId; }
    public void setGameId(String gameId) { this.gameId = gameId; }
    public int getCycle() { return cycle; }
    public void setCycle(int cycle) { this.cycle = cycle; }
    public String getEventType() { return eventType; }
    public void setEventType(String eventType) { this.eventType = eventType; }
    public String getEventData() { return eventData; }
    public void setEventData(String eventData) { this.eventData = eventData; }
    public String getCreatedAt() { return createdAt; }
    public void setCreatedAt(String createdAt) { this.createdAt = createdAt; }
}
