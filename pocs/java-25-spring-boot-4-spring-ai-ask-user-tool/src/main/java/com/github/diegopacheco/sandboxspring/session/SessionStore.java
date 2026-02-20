package com.github.diegopacheco.sandboxspring.session;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class SessionStore {

    private final Map<String, Session> sessions = new ConcurrentHashMap<>();

    public Session create() {
        Session session = new Session(UUID.randomUUID().toString());
        sessions.put(session.id, session);
        return session;
    }

    public Session get(String id) {
        Session session = sessions.get(id);
        if (session == null) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Session not found: " + id);
        }
        return session;
    }

    public void remove(String id) {
        sessions.remove(id);
    }
}
