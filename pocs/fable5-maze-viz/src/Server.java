import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;

public final class Server {

    public static void main(String[] args) throws Exception {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 8013;
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/", Server::handle);
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("maze race listening on http://localhost:" + port);
    }

    private static void handle(HttpExchange ex) throws IOException {
        String path = ex.getRequestURI().getPath();
        try {
            if (path.equals("/api/race")) {
                race(ex);
            } else if (path.equals("/src/Bfs.java") || path.equals("/src/Dfs.java")) {
                byte[] body = Files.readAllBytes(Path.of("src", path.substring(5)));
                send(ex, 200, "text/plain; charset=utf-8", body);
            } else {
                serveStatic(ex, path);
            }
        } catch (Exception e) {
            send(ex, 500, "text/plain", e.toString().getBytes(StandardCharsets.UTF_8));
        }
    }

    private static void race(HttpExchange ex) throws IOException {
        Map<String, String> q = query(ex.getRequestURI().getQuery());
        int size = clampOdd(parseInt(q.get("size"), 25), 11, 41);
        long seed = parseLong(q.get("seed"), System.nanoTime());
        Maze maze = new Maze(size, seed);
        Trace bfs = Bfs.solve(maze);
        Trace dfs = Dfs.solve(maze);
        StringBuilder json = new StringBuilder();
        json.append("{\"size\":").append(maze.size);
        json.append(",\"seed\":").append(seed);
        json.append(",\"start\":").append(maze.start);
        json.append(",\"goal\":").append(maze.goal);
        json.append(",\"open\":\"");
        for (boolean o : maze.open) json.append(o ? '1' : '0');
        json.append("\",\"bfs\":");
        appendSteps(json, bfs);
        json.append(",\"dfs\":");
        appendSteps(json, dfs);
        json.append('}');
        send(ex, 200, "application/json", json.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static void appendSteps(StringBuilder json, Trace trace) {
        json.append('[');
        for (int i = 0; i < trace.steps.size(); i++) {
            Trace.Step s = trace.steps.get(i);
            if (i > 0) json.append(',');
            json.append("[\"").append(s.type()).append("\",").append(s.cell()).append(',').append(s.line()).append(']');
        }
        json.append(']');
    }

    private static void serveStatic(HttpExchange ex, String path) throws IOException {
        if (path.equals("/")) path = "/index.html";
        Path root = Path.of("web").toAbsolutePath().normalize();
        Path file = root.resolve(path.substring(1)).normalize();
        if (!file.startsWith(root) || !Files.isRegularFile(file)) {
            send(ex, 404, "text/plain", "not found".getBytes(StandardCharsets.UTF_8));
            return;
        }
        send(ex, 200, mime(file.toString()), Files.readAllBytes(file));
    }

    private static String mime(String name) {
        if (name.endsWith(".html")) return "text/html; charset=utf-8";
        if (name.endsWith(".css")) return "text/css; charset=utf-8";
        if (name.endsWith(".js")) return "application/javascript; charset=utf-8";
        if (name.endsWith(".svg")) return "image/svg+xml";
        if (name.endsWith(".png")) return "image/png";
        return "application/octet-stream";
    }

    private static void send(HttpExchange ex, int code, String type, byte[] body) throws IOException {
        ex.getResponseHeaders().set("Content-Type", type);
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.sendResponseHeaders(code, body.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(body);
        }
    }

    private static Map<String, String> query(String raw) {
        Map<String, String> out = new HashMap<>();
        if (raw == null) return out;
        for (String pair : raw.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0) out.put(pair.substring(0, eq), pair.substring(eq + 1));
        }
        return out;
    }

    private static int parseInt(String value, int fallback) {
        try {
            return Integer.parseInt(value);
        } catch (Exception e) {
            return fallback;
        }
    }

    private static long parseLong(String value, long fallback) {
        try {
            return Long.parseLong(value);
        } catch (Exception e) {
            return fallback;
        }
    }

    private static int clampOdd(int value, int lo, int hi) {
        int v = Math.max(lo, Math.min(hi, value));
        return v % 2 == 0 ? v + 1 : v;
    }
}
