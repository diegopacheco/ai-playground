package com.github.diegopacheco.sandboxspring.event;

import org.springaicommunity.agent.tools.TodoWriteTool.Todos;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

@Component
public class TodoProgressListener {

    @EventListener
    public void onTodoUpdate(TodoUpdateEvent event) {
        int completed = (int) event.getTodos().stream()
                .filter(t -> t.status() == Todos.Status.completed)
                .count();
        int total = event.getTodos().size();
        System.out.printf("%nProgress: %d/%d tasks completed (%.0f%%)%n", completed, total, (completed * 100.0 / total));
        event.getTodos().forEach(t -> {
            String marker = switch (t.status()) {
                case completed -> "[✓]";
                case in_progress -> "[→]";
                case pending -> "[ ]";
            };
            System.out.printf("  %s %s%n", marker, t.content());
        });
    }
}
