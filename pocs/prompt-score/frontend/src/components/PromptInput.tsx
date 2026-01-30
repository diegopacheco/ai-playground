interface PromptInputProps {
  value: string
  onChange: (value: string) => void
}

function PromptInput({ value, onChange }: PromptInputProps) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder="Enter your prompt here..."
      className="w-full h-64 p-4 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 resize-none focus:outline-none focus:border-blue-500"
    />
  )
}

export default PromptInput
