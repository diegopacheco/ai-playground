package com.game.terminator.controller;

import com.game.terminator.agent.AgentRunner;
import com.game.terminator.engine.GameEngine;
import com.game.terminator.model.Game;
import com.game.terminator.repository.GameEventRepository;
import com.game.terminator.repository.GameRepository;
import com.game.terminator.sse.SseBroadcaster;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/games")
public class GameController {

    private final GameRepository gameRepository;
    private final GameEventRepository eventRepository;
    private final SseBroadcaster broadcaster;

    public GameController(GameRepository gameRepository, GameEventRepository eventRepository,
                          SseBroadcaster broadcaster) {
        this.gameRepository = gameRepository;
        this.eventRepository = eventRepository;
        this.broadcaster = broadcaster;
    }

    @PostMapping
    public Map<String, Object> createGame(@RequestBody Map<String, String> body) {
        String id = UUID.randomUUID().toString();
        String terminatorAgent = body.getOrDefault("terminator_agent", "claude");
        String terminatorModel = body.getOrDefault("terminator_model", "sonnet");
        String mosquitoAgent = body.getOrDefault("mosquito_agent", "gemini");
        String mosquitoModel = body.getOrDefault("mosquito_model", "gemini-3-flash");
        int gridSize = Integer.parseInt(body.getOrDefault("grid_size", "20"));

        Game game = new Game(id, terminatorAgent, terminatorModel,
                mosquitoAgent, mosquitoModel, gridSize, "running", Instant.now().toString());
        gameRepository.save(game);

        AgentRunner termRunner = new AgentRunner(terminatorAgent, terminatorModel);
        AgentRunner mosqRunner = new AgentRunner(mosquitoAgent, mosquitoModel);

        GameEngine engine = new GameEngine(game, termRunner, mosqRunner,
                gameRepository, eventRepository, broadcaster);
        GameEngine.register(id, engine);

        Thread.ofVirtual().name("game-" + id).start(engine);

        return Map.of("id", id, "status", "running");
    }

    @GetMapping
    public List<Game> listGames() {
        return gameRepository.findAllByOrderByCreatedAtDesc();
    }

    @GetMapping("/{id}")
    public Game getGame(@PathVariable String id) {
        return gameRepository.findById(id).orElse(null);
    }

    @PostMapping("/{id}/stop")
    public Map<String, String> stopGame(@PathVariable String id) {
        GameEngine.stopGame(id);
        return Map.of("status", "stopped");
    }

    @GetMapping(value = "/{id}/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@PathVariable String id) {
        return broadcaster.subscribe(id);
    }
}
