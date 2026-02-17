import { useState } from 'react'

interface TweetFormProps {
  onTweetCreated: () => void
}

function TweetForm({ onTweetCreated }: TweetFormProps) {
  const [username, setUsername] = useState('')
  const [content, setContent] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !content.trim()) return

    fetch('http://localhost:8080/api/tweets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, content }),
    })
      .then(() => {
        setContent('')
        onTweetCreated()
      })
      .catch(() => {})
  }

  return (
    <form className="tweet-form" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Username"
        value={username}
        onChange={e => setUsername(e.target.value)}
      />
      <textarea
        rows={3}
        placeholder="What's happening?"
        value={content}
        onChange={e => setContent(e.target.value)}
      />
      <button type="submit">Tweet</button>
    </form>
  )
}

export default TweetForm
