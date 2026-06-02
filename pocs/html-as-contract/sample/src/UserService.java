package sample;

public class UserService {
    private final UserDao userDao;

    public UserService(UserDao userDao) {
        this.userDao = userDao;
    }

    public User register(String id, String name) {
        User user = new User(id, name);
        userDao.save(user);
        return user;
    }

    public User get(String id) {
        return userDao.findById(id);
    }
}
