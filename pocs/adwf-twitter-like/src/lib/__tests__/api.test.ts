import { authApi, usersApi, tweetsApi, commentsApi } from '../api';

global.fetch = jest.fn();

const mockFetch = global.fetch as jest.MockedFunction<typeof fetch>;

beforeEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
});

describe('API Client', () => {
  describe('authApi', () => {
    describe('login', () => {
      it('should send login credentials and return auth response', async () => {
        const mockResponse = {
          token: 'test-token',
          user: { id: 1, username: 'testuser', email: 'test@test.com' },
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        const result = await authApi.login({
          email: 'test@test.com',
          password: 'password123',
        });

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/auth/login',
          expect.objectContaining({
            method: 'POST',
            headers: expect.objectContaining({
              'Content-Type': 'application/json',
            }),
            body: JSON.stringify({
              email: 'test@test.com',
              password: 'password123',
            }),
          })
        );

        expect(result).toEqual(mockResponse);
      });

      it('should throw error on failed login', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: 401,
          text: async () => 'Invalid credentials',
        } as Response);

        await expect(
          authApi.login({
            email: 'wrong@test.com',
            password: 'wrongpass',
          })
        ).rejects.toThrow();
      });
    });

    describe('register', () => {
      it('should send registration data and return auth response', async () => {
        const mockResponse = {
          token: 'new-token',
          user: { id: 2, username: 'newuser', email: 'new@test.com' },
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        } as Response);

        const result = await authApi.register({
          username: 'newuser',
          email: 'new@test.com',
          password: 'password123',
        });

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/auth/register',
          expect.objectContaining({
            method: 'POST',
            body: JSON.stringify({
              username: 'newuser',
              email: 'new@test.com',
              password: 'password123',
            }),
          })
        );

        expect(result).toEqual(mockResponse);
      });
    });

    describe('logout', () => {
      it('should call logout endpoint', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => undefined,
        } as Response);

        await authApi.logout();

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/auth/logout',
          expect.objectContaining({
            method: 'POST',
          })
        );
      });
    });
  });

  describe('usersApi', () => {
    beforeEach(() => {
      localStorage.setItem('token', 'test-token');
    });

    describe('getUser', () => {
      it('should fetch user by id with auth token', async () => {
        const mockUser = {
          id: 1,
          username: 'testuser',
          email: 'test@test.com',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockUser,
        } as Response);

        const result = await usersApi.getUser(1);

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/users/1',
          expect.objectContaining({
            headers: expect.objectContaining({
              Authorization: 'Bearer test-token',
            }),
          })
        );

        expect(result).toEqual(mockUser);
      });
    });

    describe('follow', () => {
      it('should send follow request', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => undefined,
        } as Response);

        await usersApi.follow(123);

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/users/123/follow',
          expect.objectContaining({
            method: 'POST',
          })
        );
      });
    });

    describe('unfollow', () => {
      it('should send unfollow request', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => undefined,
        } as Response);

        await usersApi.unfollow(123);

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/users/123/follow',
          expect.objectContaining({
            method: 'DELETE',
          })
        );
      });
    });
  });

  describe('tweetsApi', () => {
    beforeEach(() => {
      localStorage.setItem('token', 'test-token');
    });

    describe('createTweet', () => {
      it('should create a new tweet', async () => {
        const mockTweet = {
          id: 1,
          user_id: 1,
          content: 'Test tweet',
          created_at: '2024-01-01T00:00:00Z',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockTweet,
        } as Response);

        const result = await tweetsApi.createTweet('Test tweet');

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/tweets',
          expect.objectContaining({
            method: 'POST',
            body: JSON.stringify({ content: 'Test tweet' }),
          })
        );

        expect(result).toEqual(mockTweet);
      });
    });

    describe('likeTweet', () => {
      it('should like a tweet', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => undefined,
        } as Response);

        await tweetsApi.likeTweet(456);

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/tweets/456/like',
          expect.objectContaining({
            method: 'POST',
          })
        );
      });
    });

    describe('unlikeTweet', () => {
      it('should unlike a tweet', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => undefined,
        } as Response);

        await tweetsApi.unlikeTweet(456);

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/tweets/456/like',
          expect.objectContaining({
            method: 'DELETE',
          })
        );
      });
    });

    describe('deleteTweet', () => {
      it('should delete a tweet', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => undefined,
        } as Response);

        await tweetsApi.deleteTweet(789);

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/tweets/789',
          expect.objectContaining({
            method: 'DELETE',
          })
        );
      });
    });
  });

  describe('commentsApi', () => {
    beforeEach(() => {
      localStorage.setItem('token', 'test-token');
    });

    describe('addComment', () => {
      it('should add a comment to a tweet', async () => {
        const mockComment = {
          id: 1,
          user_id: 1,
          tweet_id: 123,
          content: 'Test comment',
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockComment,
        } as Response);

        const result = await commentsApi.addComment(123, 'Test comment');

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/tweets/123/comments',
          expect.objectContaining({
            method: 'POST',
            body: JSON.stringify({ content: 'Test comment' }),
          })
        );

        expect(result).toEqual(mockComment);
      });
    });

    describe('getComments', () => {
      it('should fetch comments for a tweet', async () => {
        const mockComments = [
          { id: 1, content: 'Comment 1' },
          { id: 2, content: 'Comment 2' },
        ];

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockComments,
        } as Response);

        const result = await commentsApi.getComments(123);

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/tweets/123/comments',
          expect.any(Object)
        );

        expect(result).toEqual(mockComments);
      });
    });

    describe('deleteComment', () => {
      it('should delete a comment', async () => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: async () => undefined,
        } as Response);

        await commentsApi.deleteComment(999);

        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/api/comments/999',
          expect.objectContaining({
            method: 'DELETE',
          })
        );
      });
    });
  });

  describe('Error Handling', () => {
    it('should throw error with response text when available', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: async () => 'Bad request error',
      } as Response);

      await expect(authApi.login({ email: 'test', password: 'test' })).rejects.toThrow(
        'Bad request error'
      );
    });

    it('should throw error with status when no text available', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => '',
      } as Response);

      await expect(authApi.login({ email: 'test', password: 'test' })).rejects.toThrow(
        'HTTP error! status: 500'
      );
    });
  });

  describe('Authorization Header', () => {
    it('should include Bearer token when available', async () => {
      localStorage.setItem('token', 'my-auth-token');

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      } as Response);

      await usersApi.getUser(1);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer my-auth-token',
          }),
        })
      );
    });

    it('should not include Authorization header when token not available', async () => {
      localStorage.removeItem('token');

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      } as Response);

      await authApi.login({ email: 'test', password: 'test' });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.not.objectContaining({
            Authorization: expect.any(String),
          }),
        })
      );
    });
  });
});
