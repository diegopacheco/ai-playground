import TweetCard from './TweetCard'

interface Tweet {
  id: number
  username: string
  content: string
  likes: number
  timestamp: string
}

interface TimelineProps {
  tweets: Tweet[]
  onLike: () => void
}

function Timeline({ tweets, onLike }: TimelineProps) {
  return (
    <div className="timeline">
      {tweets.map(tweet => (
        <TweetCard key={tweet.id} tweet={tweet} onLike={onLike} />
      ))}
    </div>
  )
}

export default Timeline
