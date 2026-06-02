package sample;

public class AccountService {
    private final UserDao userDao;

    public AccountService(UserDao userDao) {
        this.userDao = userDao;
    }

    public boolean exists(String id) {
        return userDao.findById(id) != null;
    }
}
