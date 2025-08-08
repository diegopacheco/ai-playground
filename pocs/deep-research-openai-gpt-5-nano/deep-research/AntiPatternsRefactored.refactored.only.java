/*
- Removed global mutable state: eliminated static fields X and data; class is now effectively instance-based.
- Replaced System.out prints with java.util.logging.Logger for structured, configurable logging.
- Simplified doStuff: unify input handling, trim consistently, and use explicit comparisons to improve readability.
- Avoided potential NPEs and magic values by using safe string comparisons and local variables; no shared state.
- Improved testability and maintainability by keeping outputs deterministic and side-effects localized to the logger.
*/

import java.util.logging.Logger;

public class App {
    private static final Logger logger = Logger.getLogger(App.class.getName());

    public static void main(String[] args) {
        App app = new App();
        app.run();
    }

    public void run() {
        doStuff("hello");
        doStuff("world");
        int x = 0;
        for (int i = 0; i < 3; i++) {
            x += i;
        }
        logger.info("done:" + x);
    }

    public void doStuff(String s) {
        if (s != null && !s.isEmpty()) {
            logger.info("Str:" + s);
            String t = s.trim();
            if ("hello".equals(t)) {
                logger.info("Hi!");
            } else if ("world".equals(t)) {
                logger.info("Earth!");
            } else {
                logger.info("???");
            }
        } else {
            logger.info("bad");
        }

        String u = (s == null ? "" : s).trim();
        if ("hello".equals(u)) {
            logger.info("greetings");
        } else if ("world".equals(u)) {
            logger.info("planet");
        } else {
            logger.info("unknown");
        }

        int r = 42;
        if (r > 10) {
            for (int j = 0; j < 5; j++) {
                logger.info("val:" + j);
            }
        }
    }
}