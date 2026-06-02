package sample;

import java.util.HashMap;
import java.util.Map;

public class UserDao {
    private final Map<String, User> store = new HashMap<>();

    public void save(User user) {
        store.put(user.getId(), user);
    }

    public User findById(String id) {
        if (store.containsKey(id)) {
            return store.get(id);
        }
        return null;
    }
}
