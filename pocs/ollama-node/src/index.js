import ollama from 'ollama'

try {
  const request = {
    model: 'deepseek-r1:1.5b',
    messages: [{ role: 'user', content: 'Why is the sky blue?' }],
  }
  console.log('Request:', request)

  const response = await ollama.chat(request)
  console.log('Response:', response)

} catch (error) {
  console.error('Error:', error)
}