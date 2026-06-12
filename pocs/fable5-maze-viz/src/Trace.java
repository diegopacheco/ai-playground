import java.util.ArrayList;
import java.util.List;

public final class Trace {

    public record Step(String type, int cell, int line) {}

    public final List<Step> steps = new ArrayList<>();

    public void visit(int cell) { add("visit", cell); }

    public void scan(int cell) { add("scan", cell); }

    public void frontier(int cell) { add("frontier", cell); }

    public void path(int cell) { add("path", cell); }

    private void add(String type, int cell) {
        int line = Thread.currentThread().getStackTrace()[3].getLineNumber();
        steps.add(new Step(type, cell, line));
    }
}
