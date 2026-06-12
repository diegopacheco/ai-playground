import java.util.ArrayDeque;

public final class Bfs {

    public static Trace solve(Maze maze) {
        Trace trace = new Trace();
        boolean[] seen = new boolean[maze.cells()];
        int[] parent = new int[maze.cells()];
        ArrayDeque<Integer> queue = new ArrayDeque<>();

        queue.add(maze.start);
        seen[maze.start] = true;
        trace.visit(maze.start);

        while (!queue.isEmpty()) {
            int cell = queue.poll();
            trace.scan(cell);

            if (cell == maze.goal) {
                walkBack(trace, parent, maze);
                return trace;
            }

            for (int next : maze.neighbors(cell)) {
                if (!seen[next]) {
                    seen[next] = true;
                    parent[next] = cell;
                    queue.add(next);
                    trace.frontier(next);
                }
            }
        }
        return trace;
    }

    private static void walkBack(Trace trace, int[] parent, Maze maze) {
        int cell = maze.goal;
        while (cell != maze.start) {
            trace.path(cell);
            cell = parent[cell];
        }
        trace.path(maze.start);
    }
}
