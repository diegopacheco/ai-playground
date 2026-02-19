interface Props {
  value: string
  onChange: (value: string) => void
}

export function SearchBar({ value, onChange }: Props) {
  return (
    <div className="search-bar">
      <input
        className="input search-input"
        type="search"
        placeholder="Search items..."
        value={value}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  )
}
