package com.github.diegopacheco.sandboxspring.event;

import org.springaicommunity.agent.tools.TodoWriteTool.Todos;
import org.springframework.context.ApplicationEvent;

import java.util.List;

public class TodoUpdateEvent extends ApplicationEvent {

    private final List<Todos.TodoItem> todos;

    public TodoUpdateEvent(Object source, List<Todos.TodoItem> todos) {
        super(source);
        this.todos = todos;
    }

    public List<Todos.TodoItem> getTodos() {
        return todos;
    }
}
