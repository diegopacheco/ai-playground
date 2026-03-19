package com.game.terminator.engine;

public class Mosquito {
    private final String id;
    private Position position;
    private int age;
    private boolean alive;

    public Mosquito(String id, Position position) {
        this.id = id;
        this.position = position;
        this.age = 0;
        this.alive = true;
    }

    public void tick() {
        age++;
        if (age >= 7) {
            alive = false;
        }
    }

    public String getId() { return id; }
    public Position getPosition() { return position; }
    public void setPosition(Position position) { this.position = position; }
    public int getAge() { return age; }
    public boolean isAlive() { return alive; }
    public void kill() { this.alive = false; }
}
