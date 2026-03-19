package com.game.terminator.engine;

public class Egg {
    private final String id;
    private final Position position;
    private int ticksAlive;
    private boolean hatched;
    private boolean destroyed;

    public Egg(String id, Position position) {
        this.id = id;
        this.position = position;
        this.ticksAlive = 0;
        this.hatched = false;
        this.destroyed = false;
    }

    public void tick() {
        ticksAlive++;
    }

    public boolean shouldHatch() {
        return ticksAlive >= 5 && !hatched && !destroyed;
    }

    public String getId() { return id; }
    public Position getPosition() { return position; }
    public int getTicksAlive() { return ticksAlive; }
    public boolean isHatched() { return hatched; }
    public void setHatched(boolean hatched) { this.hatched = hatched; }
    public boolean isDestroyed() { return destroyed; }
    public void destroy() { this.destroyed = true; }
    public boolean isActive() { return !hatched && !destroyed; }
}
