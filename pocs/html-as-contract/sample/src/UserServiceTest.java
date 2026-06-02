package sample;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Test;

public class UserServiceTest {

    @Test
    public void registerStoresUser() {
        UserDao dao = new UserDao();
        UserService service = new UserService(dao);
        User user = service.register("1", "Ann");
        assertNotNull(user);
        assertEquals("1", user.getId());
        assertEquals("Ann", user.getName());
    }
}
