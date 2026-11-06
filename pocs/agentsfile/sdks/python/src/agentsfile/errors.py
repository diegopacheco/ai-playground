class AgentsfileError(Exception):
    def __init__(self, line: int, col: int, message: str) -> None:
        super().__init__(f"line {line}, col {col}: {message}")
        self.line = line
        self.col = col
        self.message = message
