package com.game.terminator.sse;

import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

@Component
public class SseBroadcaster {

    private final Map<String, List<SseEmitter>> emitters = new ConcurrentHashMap<>();

    public SseEmitter subscribe(String gameId) {
        SseEmitter emitter = new SseEmitter(0L);
        emitters.computeIfAbsent(gameId, k -> new CopyOnWriteArrayList<>()).add(emitter);
        emitter.onCompletion(() -> removeEmitter(gameId, emitter));
        emitter.onTimeout(() -> removeEmitter(gameId, emitter));
        emitter.onError(e -> removeEmitter(gameId, emitter));
        return emitter;
    }

    public void broadcast(String gameId, String eventType, Object data) {
        List<SseEmitter> list = emitters.get(gameId);
        if (list == null) return;
        List<SseEmitter> dead = new java.util.ArrayList<>();
        for (SseEmitter emitter : list) {
            try {
                emitter.send(SseEmitter.event().name(eventType).data(data));
            } catch (IOException e) {
                dead.add(emitter);
            }
        }
        list.removeAll(dead);
    }

    public void complete(String gameId) {
        List<SseEmitter> list = emitters.remove(gameId);
        if (list == null) return;
        for (SseEmitter emitter : list) {
            try {
                emitter.complete();
            } catch (Exception ignored) {}
        }
    }

    private void removeEmitter(String gameId, SseEmitter emitter) {
        List<SseEmitter> list = emitters.get(gameId);
        if (list != null) {
            list.remove(emitter);
        }
    }
}
