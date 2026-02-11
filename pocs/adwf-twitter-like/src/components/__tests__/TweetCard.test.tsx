import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { TweetCard } from '../TweetCard';
import { useAuth } from '@/contexts/AuthContext';
import { tweetsApi } from '@/lib/api';
import { Tweet } from '@/types';

jest.mock('@/lib/api');
jest.mock('@/contexts/AuthContext');

const mockTweetsApi = tweetsApi as jest.Mocked<typeof tweetsApi>;
const mockUseAuth = useAuth as jest.MockedFunction<typeof useAuth>;

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{component}</BrowserRouter>
    </QueryClientProvider>
  );
};

const mockTweet: Tweet = {
  id: 1,
  user_id: 1,
  content: 'This is a test tweet',
  created_at: '2024-01-15T10:30:00Z',
  updated_at: '2024-01-15T10:30:00Z',
  user: {
    id: 1,
    username: 'testuser',
    email: 'test@test.com',
    display_name: 'Test User',
    created_at: '2024-01-01',
    updated_at: '2024-01-01',
  },
  likes_count: 5,
  retweets_count: 3,
  comments_count: 2,
  is_liked: false,
  is_retweeted: false,
};

beforeEach(() => {
  jest.clearAllMocks();
  mockUseAuth.mockReturnValue({
    user: null,
    token: null,
    login: jest.fn(),
    register: jest.fn(),
    logout: jest.fn(),
    isAuthenticated: false,
  });
});

describe('TweetCard', () => {
  describe('Rendering', () => {
    it('should render tweet content', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByText('This is a test tweet')).toBeInTheDocument();
    });

    it('should render author username', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByText('@testuser')).toBeInTheDocument();
    });

    it('should render author display name', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByText('Test User')).toBeInTheDocument();
    });

    it('should render username when display name is not available', () => {
      const tweetWithoutDisplayName = {
        ...mockTweet,
        user: { ...mockTweet.user!, display_name: undefined },
      };

      renderWithProviders(<TweetCard tweet={tweetWithoutDisplayName} />);

      expect(screen.getByText('testuser')).toBeInTheDocument();
    });

    it('should render likes count', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByText('5')).toBeInTheDocument();
    });

    it('should render retweets count', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByText('3')).toBeInTheDocument();
    });

    it('should render comments count', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByText('2')).toBeInTheDocument();
    });

    it('should show 0 when counts are undefined', () => {
      const tweetWithoutCounts = {
        ...mockTweet,
        likes_count: undefined,
        retweets_count: undefined,
        comments_count: undefined,
      };

      renderWithProviders(<TweetCard tweet={tweetWithoutCounts} />);

      const zeros = screen.getAllByText('0');
      expect(zeros.length).toBeGreaterThanOrEqual(3);
    });

    it('should format date correctly', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByText(/Jan 15, 2024/)).toBeInTheDocument();
    });

    it('should render user avatar with first letter', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByText('T')).toBeInTheDocument();
    });
  });

  describe('Like Functionality', () => {
    it('should call likeTweet API when like button is clicked', async () => {
      mockTweetsApi.likeTweet.mockResolvedValueOnce();

      renderWithProviders(<TweetCard tweet={mockTweet} />);

      const likeButton = screen.getByLabelText('Like');
      fireEvent.click(likeButton);

      await waitFor(() => {
        expect(mockTweetsApi.likeTweet).toHaveBeenCalledWith(1);
      });
    });

    it('should call unlikeTweet API when already liked', async () => {
      const likedTweet = { ...mockTweet, is_liked: true };
      mockTweetsApi.unlikeTweet.mockResolvedValueOnce();

      renderWithProviders(<TweetCard tweet={likedTweet} />);

      const unlikeButton = screen.getByLabelText('Unlike');
      fireEvent.click(unlikeButton);

      await waitFor(() => {
        expect(mockTweetsApi.unlikeTweet).toHaveBeenCalledWith(1);
      });
    });

    it('should show liked state with different styling', () => {
      const likedTweet = { ...mockTweet, is_liked: true };

      renderWithProviders(<TweetCard tweet={likedTweet} />);

      const likeButton = screen.getByLabelText('Unlike');
      expect(likeButton).toHaveClass('text-red-500');
    });

    it('should disable like button while request is pending', async () => {
      mockTweetsApi.likeTweet.mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

      renderWithProviders(<TweetCard tweet={mockTweet} />);

      const likeButton = screen.getByLabelText('Like');
      fireEvent.click(likeButton);

      expect(likeButton).toBeDisabled();
    });
  });

  describe('Retweet Functionality', () => {
    it('should call retweetTweet API when retweet button is clicked', async () => {
      mockTweetsApi.retweetTweet.mockResolvedValueOnce();

      renderWithProviders(<TweetCard tweet={mockTweet} />);

      const retweetButton = screen.getByLabelText('Retweet');
      fireEvent.click(retweetButton);

      await waitFor(() => {
        expect(mockTweetsApi.retweetTweet).toHaveBeenCalledWith(1);
      });
    });

    it('should call unretweetTweet API when already retweeted', async () => {
      const retweetedTweet = { ...mockTweet, is_retweeted: true };
      mockTweetsApi.unretweetTweet.mockResolvedValueOnce();

      renderWithProviders(<TweetCard tweet={retweetedTweet} />);

      const undoRetweetButton = screen.getByLabelText('Undo Retweet');
      fireEvent.click(undoRetweetButton);

      await waitFor(() => {
        expect(mockTweetsApi.unretweetTweet).toHaveBeenCalledWith(1);
      });
    });

    it('should show retweeted state with different styling', () => {
      const retweetedTweet = { ...mockTweet, is_retweeted: true };

      renderWithProviders(<TweetCard tweet={retweetedTweet} />);

      const retweetButton = screen.getByLabelText('Undo Retweet');
      expect(retweetButton).toHaveClass('text-green-500');
    });
  });

  describe('Delete Functionality', () => {
    it('should show delete button for tweet owner', () => {
      mockUseAuth.mockReturnValue({
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@test.com',
          created_at: '2024-01-01',
          updated_at: '2024-01-01',
        },
        token: 'token',
        login: jest.fn(),
        register: jest.fn(),
        logout: jest.fn(),
        isAuthenticated: true,
      });

      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByLabelText('Delete tweet')).toBeInTheDocument();
    });

    it('should not show delete button for other users tweets', () => {
      mockUseAuth.mockReturnValue({
        user: {
          id: 999,
          username: 'otheruser',
          email: 'other@test.com',
          created_at: '2024-01-01',
          updated_at: '2024-01-01',
        },
        token: 'token',
        login: jest.fn(),
        register: jest.fn(),
        logout: jest.fn(),
        isAuthenticated: true,
      });

      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.queryByLabelText('Delete tweet')).not.toBeInTheDocument();
    });

    it('should not show delete button when not authenticated', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.queryByLabelText('Delete tweet')).not.toBeInTheDocument();
    });

    it('should call deleteTweet API when delete button is clicked', async () => {
      mockUseAuth.mockReturnValue({
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@test.com',
          created_at: '2024-01-01',
          updated_at: '2024-01-01',
        },
        token: 'token',
        login: jest.fn(),
        register: jest.fn(),
        logout: jest.fn(),
        isAuthenticated: true,
      });

      mockTweetsApi.deleteTweet.mockResolvedValueOnce();

      renderWithProviders(<TweetCard tweet={mockTweet} />);

      const deleteButton = screen.getByLabelText('Delete tweet');
      fireEvent.click(deleteButton);

      await waitFor(() => {
        expect(mockTweetsApi.deleteTweet).toHaveBeenCalledWith(1);
      });
    });
  });

  describe('Navigation', () => {
    it('should have link to tweet detail page', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      const tweetLinks = screen.getAllByRole('link');
      const detailLink = tweetLinks.find((link) =>
        link.getAttribute('href')?.includes('/tweet/1')
      );

      expect(detailLink).toBeInTheDocument();
    });

    it('should have link to user profile', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      const profileLinks = screen.getAllByRole('link');
      const userProfileLink = profileLinks.find((link) =>
        link.getAttribute('href')?.includes('/profile/1')
      );

      expect(userProfileLink).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle tweet without user data', () => {
      const tweetWithoutUser = { ...mockTweet, user: undefined };

      renderWithProviders(<TweetCard tweet={tweetWithoutUser} />);

      expect(screen.getByText('This is a test tweet')).toBeInTheDocument();
    });

    it('should handle multiline content', () => {
      const multilineTweet = {
        ...mockTweet,
        content: 'Line 1\nLine 2\nLine 3',
      };

      renderWithProviders(<TweetCard tweet={multilineTweet} />);

      expect(screen.getByText('Line 1\nLine 2\nLine 3')).toBeInTheDocument();
    });

    it('should preserve whitespace in content', () => {
      const tweetWithSpaces = {
        ...mockTweet,
        content: 'Hello    world',
      };

      renderWithProviders(<TweetCard tweet={tweetWithSpaces} />);

      const content = screen.getByText('Hello    world');
      expect(content).toHaveClass('whitespace-pre-wrap');
    });
  });

  describe('Accessibility', () => {
    it('should have proper aria-labels on buttons', () => {
      renderWithProviders(<TweetCard tweet={mockTweet} />);

      expect(screen.getByLabelText('Like')).toBeInTheDocument();
      expect(screen.getByLabelText('Retweet')).toBeInTheDocument();
    });

    it('should change aria-label when liked', () => {
      const likedTweet = { ...mockTweet, is_liked: true };

      renderWithProviders(<TweetCard tweet={likedTweet} />);

      expect(screen.getByLabelText('Unlike')).toBeInTheDocument();
    });

    it('should change aria-label when retweeted', () => {
      const retweetedTweet = { ...mockTweet, is_retweeted: true };

      renderWithProviders(<TweetCard tweet={retweetedTweet} />);

      expect(screen.getByLabelText('Undo Retweet')).toBeInTheDocument();
    });
  });
});
