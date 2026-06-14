import { memo, useCallback, useState } from "react";
import { products } from "../data.js";

function SearchBar({ onSearch = () => {} }) {
  const [term, setTerm] = useState("");

  const handleChange = useCallback(
    (e) => {
      const next = e.target.value;
      setTerm(next);
      onSearch(next);
    },
    [onSearch]
  );

  const suggestions = term
    ? products
        .filter((p) => p.name.toLowerCase().includes(term.toLowerCase()))
        .slice(0, 5)
    : [];

  return (
    <div className="search-bar">
      <input
        className="search-bar__input"
        placeholder="Search catalog…"
        value={term}
        onChange={handleChange}
      />
      {suggestions.length > 0 && (
        <ul className="search-bar__list">
          {suggestions.map((s) => (
            <li key={s.id}>{s.name}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default memo(SearchBar);
