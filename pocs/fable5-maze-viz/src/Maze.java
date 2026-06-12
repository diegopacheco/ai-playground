import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public final class Maze {

    public final int size;
    public final boolean[] open;
    public final int start;
    public final int goal;

    public Maze(int size, long seed) {
        this.size = size;
        this.open = new boolean[size * size];
        this.start = size + 1;
        this.goal = (size - 2) * size + (size - 2);
        carve(new Random(seed));
    }

    public int cells() {
        return size * size;
    }

    public List<Integer> neighbors(int cell) {
        int x = cell % size;
        int y = cell / size;
        List<Integer> out = new ArrayList<>(4);
        if (y > 0 && open[cell - size]) out.add(cell - size);
        if (x < size - 1 && open[cell + 1]) out.add(cell + 1);
        if (y < size - 1 && open[cell + size]) out.add(cell + size);
        if (x > 0 && open[cell - 1]) out.add(cell - 1);
        return out;
    }

    private void carve(Random rnd) {
        boolean[] visited = new boolean[size * size];
        List<Integer> stack = new ArrayList<>();
        open[start] = true;
        visited[start] = true;
        stack.add(start);
        int[] dx = {0, 2, 0, -2};
        int[] dy = {-2, 0, 2, 0};
        while (!stack.isEmpty()) {
            int cell = stack.get(stack.size() - 1);
            int x = cell % size;
            int y = cell / size;
            List<Integer> dirs = new ArrayList<>(List.of(0, 1, 2, 3));
            Collections.shuffle(dirs, rnd);
            boolean moved = false;
            for (int d : dirs) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                if (nx < 1 || ny < 1 || nx >= size - 1 || ny >= size - 1) continue;
                int next = ny * size + nx;
                if (visited[next]) continue;
                int wall = (y + dy[d] / 2) * size + (x + dx[d] / 2);
                open[wall] = true;
                open[next] = true;
                visited[next] = true;
                stack.add(next);
                moved = true;
                break;
            }
            if (!moved) stack.remove(stack.size() - 1);
        }
    }
}
