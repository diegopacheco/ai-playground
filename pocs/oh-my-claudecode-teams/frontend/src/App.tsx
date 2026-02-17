import { useState, useEffect } from 'react'
import './App.css'
import TweetForm from './TweetForm'
import Timeline from './Timeline'

interface Tweet {
  id: number
  username: string
  content: string
  likes: number
  timestamp: string
}

const API_URL = 'http://localhost:8080/api/tweets'

function App() {
  const [tweets, setTweets] = useState<Tweet[]>([])

  const fetchTweets = () => {
    fetch(API_URL)
      .then(res => res.json())
      .then(data => setTweets(data))
      .catch(() => {})
  }

  useEffect(() => {
    fetchTweets()
    const interval = setInterval(fetchTweets, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="app">
      <div className="app-header">Twitter Clone</div>
      <TweetForm onTweetCreated={fetchTweets} />
      <Timeline tweets={tweets} onLike={fetchTweets} />
    </div>
  )
}

export default App
