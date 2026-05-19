// ListCard.jsx — generic emoji-headed list. Used for skills, certifications,
// papers, feature POCs, and the 4-col AI POCs grid.

function ListCard({ id, heading, headingLevel = 'h3', items, listClass = 'skills-list' }) {
  const H = headingLevel;
  return (
    <article id={id}>
      <H>{heading}</H>
      <ul className={listClass}>
        {items.map((it, i) => (
          <li key={i}>
            {it.href ? (
              <a href={it.href} target="_blank" rel="noopener noreferrer">{it.label}</a>
            ) : (
              it.label
            )}
          </li>
        ))}
      </ul>
    </article>
  );
}

window.ListCard = ListCard;
