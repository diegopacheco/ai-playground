import java.util.ArrayDeque;

public final class Dfs {

    public static Trace solve(Maze maze) {
        Trace trace = new Trace();
        boolean[] seen = new boolean[maze.cells()];
        int[] parent = new int[maze.cells()];
        ArrayDeque<Integer> stack = new ArrayDeque<>();

        parent[maze.start] = maze.start;
        stack.push(maze.start);

        while (!stack.isEmpty()) {
            int cell = stack.pop();
            if (seen[cell]) continue;
            seen[cell] = true;
            trace.scan(cell);

            if (cell == maze.goal) {
                walkBack(trace, parent, maze);
                return trace;
            }

            for (int next : maze.neighbors(cell)) {
                if (!seen[next]) {
                    parent[next] = cell;
                    stack.push(next);
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
