package io.agentsfile;

public enum TokenKind {
    IDENTIFIER("identifier"),
    STRING("string"),
    EQUALS("'='"),
    LBRACE("'{'"),
    RBRACE("'}'"),
    LBRACKET("'['"),
    RBRACKET("']'"),
    COMMA("','"),
    EOF("end of file");

    private final String label;

    TokenKind(String label) {
        this.label = label;
    }

    public String label() {
        return label;
    }
}
