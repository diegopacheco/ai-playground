package com.game.terminator.engine;

import com.game.terminator.agent.AgentRunner;
import com.game.terminator.model.Game;
import com.game.terminator.model.GameEvent;
import com.game.terminator.repository.GameEventRepository;
import com.game.terminator.repository.GameRepository;
import com.game.terminator.sse.SseBroadcaster;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

public class GameEngine implements Runnable {

    private final Game game;
    private final AgentRunner terminatorAgent;
    private final AgentRunner mosquitoAgent;
    private final GameRepository gameRepository;
    private final GameEventRepository eventRepository;
    private final SseBroadcaster broadcaster;
    private final int gridSize;

    private Position terminatorPos;
    private final List<Mosquito> mosquitos = Collections.synchronizedList(new ArrayList<>());
    private final List<Egg> eggs = Collections.synchronizedList(new ArrayList<>());
    private int cycle = 0;
    private int totalKills = 0;
    private int totalHatched = 0;
    private int totalDates = 0;
    private int maxMosquitos = 0;
    private int mosquitoCounter = 0;
    private int eggCounter = 0;
    private volatile boolean running = true;

    private static final Map<String, GameEngine> RUNNING_GAMES = new ConcurrentHashMap<>();

    public GameEngine(Game game, AgentRunner terminatorAgent, AgentRunner mosquitoAgent,
                      GameRepository gameRepository, GameEventRepository eventRepository,
                      SseBroadcaster broadcaster) {
        this.game = game;
        this.terminatorAgent = terminatorAgent;
        this.mosquitoAgent = mosquitoAgent;
        this.gameRepository = gameRepository;
        this.eventRepository = eventRepository;
        this.broadcaster = broadcaster;
        this.gridSize = game.getGridSize();
    }

    public static void register(String gameId, GameEngine engine) {
        RUNNING_GAMES.put(gameId, engine);
    }

    public static void stopGame(String gameId) {
        GameEngine engine = RUNNING_GAMES.get(gameId);
        if (engine != null) {
            engine.running = false;
        }
    }

    @Override
    public void run() {
        initializeEntities();
        broadcastGameStart();

        while (running) {
            cycle++;
            try {
                Direction termDir = getTerminatorMove();
                terminatorPos = terminatorPos.move(termDir, gridSize);

                moveMosquitos();

                List<Map<String, Object>> killEvents = processTerminatorKills();

                List<Map<String, Object>> dateEvents = processMosquitoDating();

                List<Map<String, Object>> hatchEvents = processEggHatching();

                List<String> deathEvents = processMosquitoAging();

                int aliveMosquitos = (int) mosquitos.stream().filter(Mosquito::isAlive).count();
                long activeEggs = eggs.stream().filter(Egg::isActive).count();
                maxMosquitos = Math.max(maxMosquitos, aliveMosquitos);

                broadcastCycleUpdate(killEvents, dateEvents, hatchEvents, deathEvents);

                String winner = checkWinCondition(aliveMosquitos, activeEggs);
                if (winner != null) {
                    endGame(winner);
                    return;
                }

                Thread.sleep(700);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                endGame("interrupted");
                return;
            }
        }
        endGame("stopped");
    }

    private void initializeEntities() {
        terminatorPos = new Position(gridSize / 2, gridSize / 2);
        for (int i = 0; i < 3; i++) {
            Position pos = randomPosition();
            while (pos.equals(terminatorPos)) {
                pos = randomPosition();
            }
            mosquitos.add(new Mosquito("m_" + (++mosquitoCounter), pos));
        }
        maxMosquitos = 3;
    }

    private Position randomPosition() {
        return new Position(
            ThreadLocalRandom.current().nextInt(gridSize),
            ThreadLocalRandom.current().nextInt(gridSize)
        );
    }

    private Direction getTerminatorMove() {
        Position nearest = findNearestTarget();
        if (nearest == null) {
            return Direction.CARDINAL[ThreadLocalRandom.current().nextInt(4)];
        }
        int dx = nearest.x() - terminatorPos.x();
        int dy = nearest.y() - terminatorPos.y();
        if (Math.abs(dx) >= Math.abs(dy)) {
            return dx > 0 ? Direction.RIGHT : Direction.LEFT;
        } else {
            return dy > 0 ? Direction.DOWN : Direction.UP;
        }
    }

    private Position findNearestTarget() {
        Position best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Mosquito m : mosquitos) {
            if (!m.isAlive()) continue;
            int d = Math.abs(m.getPosition().x() - terminatorPos.x()) + Math.abs(m.getPosition().y() - terminatorPos.y());
            if (d < bestDist) { bestDist = d; best = m.getPosition(); }
        }
        for (Egg e : eggs) {
            if (!e.isActive()) continue;
            int d = Math.abs(e.getPosition().x() - terminatorPos.x()) + Math.abs(e.getPosition().y() - terminatorPos.y());
            if (d < bestDist) { bestDist = d; best = e.getPosition(); }
        }
        return best;
    }

    private void moveMosquitos() {
        List<Mosquito> alive = mosquitos.stream().filter(Mosquito::isAlive).toList();
        if (alive.isEmpty()) return;

        for (Mosquito m : alive) {
            Direction dir = evadeTerminator(m);
            m.setPosition(m.getPosition().move(dir, gridSize));
        }
    }

    private Direction evadeTerminator(Mosquito m) {
        int dx = m.getPosition().x() - terminatorPos.x();
        int dy = m.getPosition().y() - terminatorPos.y();
        int dist = Math.abs(dx) + Math.abs(dy);
        if (dist <= 4) {
            if (Math.abs(dx) >= Math.abs(dy)) {
                return dx > 0 ? Direction.RIGHT : Direction.LEFT;
            } else {
                return dy > 0 ? Direction.DOWN : Direction.UP;
            }
        }
        List<Mosquito> others = mosquitos.stream()
            .filter(Mosquito::isAlive)
            .filter(o -> !o.getId().equals(m.getId()))
            .toList();
        if (!others.isEmpty() && ThreadLocalRandom.current().nextInt(3) == 0) {
            Mosquito target = others.get(ThreadLocalRandom.current().nextInt(others.size()));
            int tdx = target.getPosition().x() - m.getPosition().x();
            int tdy = target.getPosition().y() - m.getPosition().y();
            if (tdx > 0 && tdy > 0) return Direction.DOWN_RIGHT;
            if (tdx > 0 && tdy < 0) return Direction.UP_RIGHT;
            if (tdx < 0 && tdy > 0) return Direction.DOWN_LEFT;
            if (tdx < 0 && tdy < 0) return Direction.UP_LEFT;
            if (tdx > 0) return Direction.RIGHT;
            if (tdx < 0) return Direction.LEFT;
            if (tdy > 0) return Direction.DOWN;
            return Direction.UP;
        }
        return Direction.ALL[ThreadLocalRandom.current().nextInt(Direction.ALL.length)];
    }

    private Direction randomDirection() {
        return Direction.ALL[ThreadLocalRandom.current().nextInt(Direction.ALL.length)];
    }

    private List<Map<String, Object>> processTerminatorKills() {
        List<Map<String, Object>> events = new ArrayList<>();
        List<Mosquito> killed = mosquitos.stream()
            .filter(Mosquito::isAlive)
            .filter(m -> m.getPosition().equals(terminatorPos))
            .toList();

        List<Egg> destroyedEggs = eggs.stream()
            .filter(Egg::isActive)
            .filter(e -> e.getPosition().equals(terminatorPos))
            .toList();

        if (!killed.isEmpty() || !destroyedEggs.isEmpty()) {
            killed.forEach(Mosquito::kill);
            destroyedEggs.forEach(Egg::destroy);
            totalKills += killed.size() + destroyedEggs.size();

            Map<String, Object> event = new HashMap<>();
            event.put("position", Map.of("x", terminatorPos.x(), "y", terminatorPos.y()));
            event.put("killed_mosquitos", killed.stream().map(Mosquito::getId).toList());
            event.put("killed_eggs", destroyedEggs.stream().map(Egg::getId).toList());
            events.add(event);

            saveEvent("terminator_kill", event);
        }
        return events;
    }

    private List<Map<String, Object>> processMosquitoDating() {
        List<Map<String, Object>> events = new ArrayList<>();
        List<Mosquito> alive = mosquitos.stream().filter(Mosquito::isAlive).toList();

        Map<Position, List<Mosquito>> byPosition = alive.stream()
            .collect(Collectors.groupingBy(Mosquito::getPosition));

        for (Map.Entry<Position, List<Mosquito>> entry : byPosition.entrySet()) {
            List<Mosquito> group = entry.getValue();
            if (group.size() >= 2) {
                String eggId = "e_" + (++eggCounter);
                eggs.add(new Egg(eggId, entry.getKey()));
                totalDates++;

                Map<String, Object> event = new HashMap<>();
                event.put("position", Map.of("x", entry.getKey().x(), "y", entry.getKey().y()));
                event.put("mosquito_ids", group.stream().map(Mosquito::getId).limit(2).toList());
                event.put("egg_id", eggId);
                events.add(event);

                saveEvent("mosquito_date", event);
            }
        }
        return events;
    }

    private List<Map<String, Object>> processEggHatching() {
        List<Map<String, Object>> events = new ArrayList<>();
        for (Egg egg : eggs) {
            if (!egg.isActive()) continue;
            egg.tick();
            if (egg.shouldHatch()) {
                egg.setHatched(true);
                String mId = "m_" + (++mosquitoCounter);
                mosquitos.add(new Mosquito(mId, egg.getPosition()));
                totalHatched++;

                Map<String, Object> event = new HashMap<>();
                event.put("position", Map.of("x", egg.getPosition().x(), "y", egg.getPosition().y()));
                event.put("egg_id", egg.getId());
                event.put("new_mosquito_id", mId);
                events.add(event);

                saveEvent("egg_hatch", event);
            }
        }
        return events;
    }

    private List<String> processMosquitoAging() {
        List<String> deaths = new ArrayList<>();
        for (Mosquito m : mosquitos) {
            if (!m.isAlive()) continue;
            m.tick();
            if (!m.isAlive()) {
                deaths.add(m.getId());
                saveEvent("mosquito_death", Map.of("mosquito_id", m.getId(), "cause", "age"));
            }
        }
        return deaths;
    }

    private String checkWinCondition(int aliveMosquitos, long activeEggs) {
        if (aliveMosquitos == 0 && activeEggs == 0 && cycle > 1) {
            return "terminator";
        }
        if (aliveMosquitos >= 50) {
            return "mosquitos";
        }
        if (cycle >= 200) {
            return "draw";
        }
        return null;
    }

    private void endGame(String winner) {
        running = false;
        RUNNING_GAMES.remove(game.getId());
        game.setWinner(winner);
        game.setTotalCycles(cycle);
        game.setMaxMosquitos(maxMosquitos);
        game.setTotalKills(totalKills);
        game.setTotalHatched(totalHatched);
        game.setTotalDates(totalDates);
        game.setStatus("finished");
        game.setEndedAt(Instant.now().toString());
        gameRepository.save(game);

        int aliveMosquitos = (int) mosquitos.stream().filter(Mosquito::isAlive).count();
        long activeEggs = eggs.stream().filter(Egg::isActive).count();

        broadcaster.broadcast(game.getId(), "game_over", Map.of(
            "winner", winner,
            "cycles", cycle,
            "total_kills", totalKills,
            "total_hatched", totalHatched,
            "total_dates", totalDates,
            "max_mosquitos", maxMosquitos,
            "alive_mosquitos", aliveMosquitos,
            "active_eggs", activeEggs
        ));
        broadcaster.complete(game.getId());
    }

    private void broadcastGameStart() {
        List<Map<String, Object>> mosquitoData = mosquitos.stream().map(m -> Map.<String, Object>of(
            "id", m.getId(),
            "x", m.getPosition().x(),
            "y", m.getPosition().y()
        )).toList();

        broadcaster.broadcast(game.getId(), "game_start", Map.of(
            "grid_size", gridSize,
            "terminator", Map.of("x", terminatorPos.x(), "y", terminatorPos.y()),
            "mosquitos", mosquitoData,
            "terminator_agent", terminatorAgent.getName() + "/" + terminatorAgent.getModel(),
            "mosquito_agent", mosquitoAgent.getName() + "/" + mosquitoAgent.getModel()
        ));
    }

    private void broadcastCycleUpdate(List<Map<String, Object>> kills,
                                       List<Map<String, Object>> dates,
                                       List<Map<String, Object>> hatches,
                                       List<String> deaths) {
        List<Mosquito> alive = mosquitos.stream().filter(Mosquito::isAlive).toList();
        List<Egg> active = eggs.stream().filter(Egg::isActive).toList();

        List<Map<String, Object>> mosquitoData = alive.stream().map(m -> {
            Map<String, Object> map = new HashMap<>();
            map.put("id", m.getId());
            map.put("x", m.getPosition().x());
            map.put("y", m.getPosition().y());
            map.put("age", m.getAge());
            return map;
        }).toList();

        List<Map<String, Object>> eggData = active.stream().map(e -> {
            Map<String, Object> map = new HashMap<>();
            map.put("id", e.getId());
            map.put("x", e.getPosition().x());
            map.put("y", e.getPosition().y());
            map.put("ticks", e.getTicksAlive());
            return map;
        }).toList();

        Map<String, Object> update = new HashMap<>();
        update.put("cycle", cycle);
        update.put("terminator", Map.of("x", terminatorPos.x(), "y", terminatorPos.y()));
        update.put("mosquitos", mosquitoData);
        update.put("eggs", eggData);
        update.put("kills", kills);
        update.put("dates", dates);
        update.put("hatches", hatches);
        update.put("deaths", deaths);
        update.put("total_kills", totalKills);
        update.put("total_hatched", totalHatched);
        update.put("total_dates", totalDates);
        update.put("alive_count", alive.size());
        update.put("egg_count", active.size());

        broadcaster.broadcast(game.getId(), "cycle_update", update);
    }

    private String buildGridStatePrompt(String role) {
        List<Mosquito> alive = mosquitos.stream().filter(Mosquito::isAlive).toList();
        List<Egg> active = eggs.stream().filter(Egg::isActive).toList();

        StringBuilder sb = new StringBuilder();
        sb.append("You are playing a grid game on a ").append(gridSize).append("x").append(gridSize).append(" grid.\n");

        if ("terminator".equals(role)) {
            sb.append("You control the Terminator at position (").append(terminatorPos.x()).append(",").append(terminatorPos.y()).append(").\n");
            sb.append("You can move: UP, DOWN, LEFT, RIGHT (one step).\n");
            sb.append("Your goal: kill all mosquitos and eggs by stepping on them.\n");
        } else {
            sb.append("You control the mosquitos. Terminator is at (").append(terminatorPos.x()).append(",").append(terminatorPos.y()).append(").\n");
            sb.append("Each mosquito can move: UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT.\n");
            sb.append("Your goal: survive and breed. When 2 mosquitos share a cell they date and lay an egg.\n");
        }

        sb.append("Mosquitos: ");
        for (Mosquito m : alive) {
            sb.append(m.getId()).append("(").append(m.getPosition().x()).append(",").append(m.getPosition().y()).append(" age:").append(m.getAge()).append(") ");
        }
        sb.append("\nEggs: ");
        for (Egg e : active) {
            sb.append(e.getId()).append("(").append(e.getPosition().x()).append(",").append(e.getPosition().y()).append(" ticks:").append(e.getTicksAlive()).append(") ");
        }

        if ("terminator".equals(role)) {
            sb.append("\nRespond ONLY with JSON: {\"direction\": \"UP|DOWN|LEFT|RIGHT\"}");
        } else {
            sb.append("\nRespond ONLY with JSON: {\"moves\": [{\"id\": \"m_1\", \"direction\": \"UP\"}, ...]}");
        }

        return sb.toString();
    }

    private Direction parseTerminatorDirection(String response) {
        if (response == null || response.isEmpty()) return null;
        try {
            int idx = response.indexOf("\"direction\"");
            if (idx < 0) return null;
            int colon = response.indexOf(":", idx);
            int quote1 = response.indexOf("\"", colon + 1);
            int quote2 = response.indexOf("\"", quote1 + 1);
            if (quote1 < 0 || quote2 < 0) return null;
            String dir = response.substring(quote1 + 1, quote2).trim();
            Direction d = Direction.fromString(dir);
            if (d != null && List.of(Direction.CARDINAL).contains(d)) {
                return d;
            }
        } catch (Exception e) {
            return null;
        }
        return null;
    }

    private Map<String, Direction> parseMosquitoMoves(String response, List<Mosquito> alive) {
        Map<String, Direction> moves = new HashMap<>();
        if (response == null || response.isEmpty()) return moves;
        try {
            int idx = 0;
            while (true) {
                int idIdx = response.indexOf("\"id\"", idx);
                if (idIdx < 0) break;
                int colon1 = response.indexOf(":", idIdx);
                int q1 = response.indexOf("\"", colon1 + 1);
                int q2 = response.indexOf("\"", q1 + 1);
                String mId = response.substring(q1 + 1, q2);

                int dirIdx = response.indexOf("\"direction\"", q2);
                if (dirIdx < 0) break;
                int colon2 = response.indexOf(":", dirIdx);
                int q3 = response.indexOf("\"", colon2 + 1);
                int q4 = response.indexOf("\"", q3 + 1);
                String dir = response.substring(q3 + 1, q4);

                Direction d = Direction.fromString(dir);
                if (d != null) {
                    moves.put(mId, d);
                }
                idx = q4 + 1;
            }
        } catch (Exception e) {
            return moves;
        }
        return moves;
    }

    private void saveEvent(String eventType, Object data) {
        try {
            String json = data.toString();
            eventRepository.save(new GameEvent(game.getId(), cycle, eventType, json, Instant.now().toString()));
        } catch (Exception ignored) {}
    }
}
