import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TweetComposer } from '../TweetComposer';
import { tweetsApi } from '@/lib/api';

jest.mock('@/lib/api');

const mockTweetsApi = tweetsApi as jest.Mocked<typeof tweetsApi>;

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>
  );
};

beforeEach(() => {
  jest.clearAllMocks();
});

describe('TweetComposer', () => {
  describe('Rendering', () => {
    it('should render textarea and submit button', () => {
      renderWithQueryClient(<TweetComposer />);

      expect(screen.getByPlaceholderText("What's happening?")).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /tweet/i })).toBeInTheDocument();
    });

    it('should show character count', () => {
      renderWithQueryClient(<TweetComposer />);

      expect(screen.getByText('0/280')).toBeInTheDocument();
    });

    it('should have submit button disabled initially', () => {
      renderWithQueryClient(<TweetComposer />);

      const button = screen.getByRole('button', { name: /tweet/i });
      expect(button).toBeDisabled();
    });
  });

  describe('User Input', () => {
    it('should update character count when typing', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      fireEvent.change(textarea, { target: { value: 'Hello world' } });

      expect(screen.getByText('11/280')).toBeInTheDocument();
    });

    it('should enable submit button when content is valid', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      fireEvent.change(textarea, { target: { value: 'Valid tweet' } });

      expect(button).not.toBeDisabled();
    });

    it('should disable submit button for empty content', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      fireEvent.change(textarea, { target: { value: '' } });

      expect(button).toBeDisabled();
    });

    it('should disable submit button for whitespace-only content', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      fireEvent.change(textarea, { target: { value: '   ' } });

      expect(button).toBeDisabled();
    });

    it('should disable submit button when exceeding 280 characters', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      fireEvent.change(textarea, { target: { value: 'a'.repeat(281) } });

      expect(button).toBeDisabled();
    });

    it('should show red character count when exceeding limit', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      fireEvent.change(textarea, { target: { value: 'a'.repeat(281) } });

      const charCount = screen.getByText('281/280');
      expect(charCount).toHaveClass('text-red-500');
    });

    it('should accept exactly 280 characters', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      fireEvent.change(textarea, { target: { value: 'a'.repeat(280) } });

      expect(screen.getByText('280/280')).toBeInTheDocument();
      expect(button).not.toBeDisabled();
    });
  });

  describe('Tweet Submission', () => {
    it('should call createTweet API on form submit', async () => {
      const mockTweet = {
        id: 1,
        user_id: 1,
        content: 'Test tweet',
        created_at: '2024-01-01',
        updated_at: '2024-01-01',
      };

      mockTweetsApi.createTweet.mockResolvedValueOnce(mockTweet);

      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      fireEvent.change(textarea, { target: { value: 'Test tweet' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockTweetsApi.createTweet).toHaveBeenCalledWith('Test tweet');
      });
    });

    it('should clear content after successful submission', async () => {
      const mockTweet = {
        id: 1,
        user_id: 1,
        content: 'Test tweet',
        created_at: '2024-01-01',
        updated_at: '2024-01-01',
      };

      mockTweetsApi.createTweet.mockResolvedValueOnce(mockTweet);

      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText(
        "What's happening?"
      ) as HTMLTextAreaElement;

      fireEvent.change(textarea, { target: { value: 'Test tweet' } });
      fireEvent.submit(textarea.closest('form')!);

      await waitFor(() => {
        expect(textarea.value).toBe('');
      });
    });

    it('should show loading state during submission', async () => {
      mockTweetsApi.createTweet.mockImplementation(
        () => new Promise((resolve) => setTimeout(resolve, 100))
      );

      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      fireEvent.change(textarea, { target: { value: 'Test tweet' } });
      fireEvent.click(button);

      expect(screen.getByText('Posting...')).toBeInTheDocument();
      expect(button).toBeDisabled();
    });

    it('should show error message on failed submission', async () => {
      mockTweetsApi.createTweet.mockRejectedValueOnce(new Error('API Error'));

      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      fireEvent.change(textarea, { target: { value: 'Test tweet' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(
          screen.getByText('Failed to post tweet. Please try again.')
        ).toBeInTheDocument();
      });
    });

    it('should not submit when content is empty', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      fireEvent.submit(textarea.closest('form')!);

      expect(mockTweetsApi.createTweet).not.toHaveBeenCalled();
    });

    it('should trim whitespace before checking validity', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      fireEvent.change(textarea, { target: { value: '  Valid tweet  ' } });

      expect(button).not.toBeDisabled();
    });
  });

  describe('Form Reset', () => {
    it('should reset character count after submission', async () => {
      const mockTweet = {
        id: 1,
        user_id: 1,
        content: 'Test',
        created_at: '2024-01-01',
        updated_at: '2024-01-01',
      };

      mockTweetsApi.createTweet.mockResolvedValueOnce(mockTweet);

      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      fireEvent.change(textarea, { target: { value: 'Test tweet' } });

      expect(screen.getByText('10/280')).toBeInTheDocument();

      fireEvent.submit(textarea.closest('form')!);

      await waitFor(() => {
        expect(screen.getByText('0/280')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('should have proper aria-label on textarea', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByLabelText('Tweet content');
      expect(textarea).toBeInTheDocument();
    });

    it('should support keyboard navigation', () => {
      renderWithQueryClient(<TweetComposer />);

      const textarea = screen.getByPlaceholderText("What's happening?");
      const button = screen.getByRole('button', { name: /tweet/i });

      textarea.focus();
      expect(document.activeElement).toBe(textarea);

      button.focus();
      expect(document.activeElement).toBe(button);
    });
  });
});
