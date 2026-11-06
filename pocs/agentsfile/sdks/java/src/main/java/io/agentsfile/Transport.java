package io.agentsfile;

import java.util.Arrays;
import java.util.Optional;

public enum Transport {
    STDIO("stdio"),
    HTTP("http"),
    SSE("sse");

    private final String keyword;

    Transport(String keyword) {
        this.keyword = keyword;
    }

    public String keyword() {
        return keyword;
    }

    public static Optional<Transport> fromKeyword(String keyword) {
        return Arrays.stream(values()).filter(t -> t.keyword.equals(keyword)).findFirst();
    }
}
