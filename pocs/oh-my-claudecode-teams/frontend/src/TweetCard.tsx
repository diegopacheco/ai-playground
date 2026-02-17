interface Tweet {
  id: number
  username: string
  content: string
  likes: number
  timestamp: string
}

interface TweetCardProps {
  tweet: Tweet
  onLike: () => void
}

function TweetCard({ tweet, onLike }: TweetCardProps) {
  const handleLike = () => {
    fetch(`http://localhost:8080/api/tweets/${tweet.id}/like`, { method: 'POST' })
      .then(() => onLike())
      .catch(() => {})
  }

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleString()
  }

  return (
    <div className="tweet-card">
      <div className="tweet-card-header">
        <span className="tweet-username">@{tweet.username}</span>
        <span className="tweet-timestamp">{formatTime(tweet.timestamp)}</span>
      </div>
      <div className="tweet-content">{tweet.content}</div>
      <div className="tweet-actions">
        <button className="like-button" onClick={handleLike}>
          ♥ {tweet.likes}
        </button>
      </div>
    </div>
  )
}

export default TweetCard
