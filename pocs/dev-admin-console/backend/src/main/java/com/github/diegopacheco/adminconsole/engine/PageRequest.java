package com.github.diegopacheco.adminconsole.engine;

public record PageRequest(int size, String cursor, int pageNumber) {
    public static PageRequest first(int size) {
        return new PageRequest(size, null, 1);
    }

    public boolean isFirst() {
        return cursor == null || cursor.isBlank();
    }
}
