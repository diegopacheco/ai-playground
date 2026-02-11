import { renderHook, act, waitFor } from '@testing-library/react';
import { AuthProvider, useAuth } from '../AuthContext';
import { authApi } from '@/lib/api';

jest.mock('@/lib/api');

const mockAuthApi = authApi as jest.Mocked<typeof authApi>;

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <AuthProvider>{children}</AuthProvider>
);

beforeEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
});

describe('AuthContext', () => {
  describe('useAuth', () => {
    it('should throw error when used outside AuthProvider', () => {
      const consoleError = jest.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useAuth());
      }).toThrow('useAuth must be used within an AuthProvider');

      consoleError.mockRestore();
    });

    it('should provide auth context when used within AuthProvider', () => {
      const { result } = renderHook(() => useAuth(), { wrapper });

      expect(result.current).toBeDefined();
      expect(result.current.user).toBeNull();
      expect(result.current.token).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('Initial State', () => {
    it('should start with null user and token', () => {
      const { result } = renderHook(() => useAuth(), { wrapper });

      expect(result.current.user).toBeNull();
      expect(result.current.token).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
    });

    it('should restore user and token from localStorage', () => {
      const mockUser = {
        id: 1,
        username: 'testuser',
        email: 'test@test.com',
        created_at: '2024-01-01',
        updated_at: '2024-01-01',
      };
      const mockToken = 'stored-token';

      localStorage.setItem('token', mockToken);
      localStorage.setItem('user', JSON.stringify(mockUser));

      const { result } = renderHook(() => useAuth(), { wrapper });

      waitFor(() => {
        expect(result.current.token).toBe(mockToken);
        expect(result.current.user).toEqual(mockUser);
        expect(result.current.isAuthenticated).toBe(true);
      });
    });
  });

  describe('login', () => {
    it('should set user and token on successful login', async () => {
      const mockResponse = {
        token: 'login-token',
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@test.com',
          created_at: '2024-01-01',
          updated_at: '2024-01-01',
        },
      };

      mockAuthApi.login.mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useAuth(), { wrapper });

      await act(async () => {
        await result.current.login({
          email: 'test@test.com',
          password: 'password123',
        });
      });

      expect(result.current.token).toBe(mockResponse.token);
      expect(result.current.user).toEqual(mockResponse.user);
      expect(result.current.isAuthenticated).toBe(true);
      expect(localStorage.setItem).toHaveBeenCalledWith('token', mockResponse.token);
      expect(localStorage.setItem).toHaveBeenCalledWith(
        'user',
        JSON.stringify(mockResponse.user)
      );
    });

    it('should throw error on failed login', async () => {
      mockAuthApi.login.mockRejectedValueOnce(new Error('Invalid credentials'));

      const { result } = renderHook(() => useAuth(), { wrapper });

      await expect(
        act(async () => {
          await result.current.login({
            email: 'wrong@test.com',
            password: 'wrongpass',
          });
        })
      ).rejects.toThrow('Invalid credentials');

      expect(result.current.user).toBeNull();
      expect(result.current.token).toBeNull();
    });
  });

  describe('register', () => {
    it('should set user and token on successful registration', async () => {
      const mockResponse = {
        token: 'register-token',
        user: {
          id: 2,
          username: 'newuser',
          email: 'new@test.com',
          created_at: '2024-01-01',
          updated_at: '2024-01-01',
        },
      };

      mockAuthApi.register.mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useAuth(), { wrapper });

      await act(async () => {
        await result.current.register({
          username: 'newuser',
          email: 'new@test.com',
          password: 'password123',
        });
      });

      expect(result.current.token).toBe(mockResponse.token);
      expect(result.current.user).toEqual(mockResponse.user);
      expect(result.current.isAuthenticated).toBe(true);
      expect(localStorage.setItem).toHaveBeenCalledWith('token', mockResponse.token);
      expect(localStorage.setItem).toHaveBeenCalledWith(
        'user',
        JSON.stringify(mockResponse.user)
      );
    });

    it('should throw error on failed registration', async () => {
      mockAuthApi.register.mockRejectedValueOnce(
        new Error('Username already exists')
      );

      const { result } = renderHook(() => useAuth(), { wrapper });

      await expect(
        act(async () => {
          await result.current.register({
            username: 'taken',
            email: 'test@test.com',
            password: 'password123',
          });
        })
      ).rejects.toThrow('Username already exists');

      expect(result.current.user).toBeNull();
      expect(result.current.token).toBeNull();
    });
  });

  describe('logout', () => {
    it('should clear user and token on logout', async () => {
      const mockResponse = {
        token: 'test-token',
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@test.com',
          created_at: '2024-01-01',
          updated_at: '2024-01-01',
        },
      };

      mockAuthApi.login.mockResolvedValueOnce(mockResponse);
      mockAuthApi.logout.mockResolvedValueOnce();

      const { result } = renderHook(() => useAuth(), { wrapper });

      await act(async () => {
        await result.current.login({
          email: 'test@test.com',
          password: 'password123',
        });
      });

      expect(result.current.isAuthenticated).toBe(true);

      act(() => {
        result.current.logout();
      });

      expect(result.current.user).toBeNull();
      expect(result.current.token).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
      expect(localStorage.removeItem).toHaveBeenCalledWith('token');
      expect(localStorage.removeItem).toHaveBeenCalledWith('user');
    });

    it('should clear state even if logout API call fails', () => {
      mockAuthApi.logout.mockRejectedValueOnce(new Error('Network error'));

      const { result } = renderHook(() => useAuth(), { wrapper });

      localStorage.setItem('token', 'test-token');
      localStorage.setItem('user', JSON.stringify({ id: 1 }));

      act(() => {
        result.current.logout();
      });

      expect(result.current.user).toBeNull();
      expect(result.current.token).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('isAuthenticated', () => {
    it('should return false when no token', () => {
      const { result } = renderHook(() => useAuth(), { wrapper });

      expect(result.current.isAuthenticated).toBe(false);
    });

    it('should return true when token exists', async () => {
      const mockResponse = {
        token: 'test-token',
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@test.com',
          created_at: '2024-01-01',
          updated_at: '2024-01-01',
        },
      };

      mockAuthApi.login.mockResolvedValueOnce(mockResponse);

      const { result } = renderHook(() => useAuth(), { wrapper });

      await act(async () => {
        await result.current.login({
          email: 'test@test.com',
          password: 'password123',
        });
      });

      expect(result.current.isAuthenticated).toBe(true);
    });
  });
});
