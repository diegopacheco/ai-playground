// SocialsGrid.jsx — 13 social-platform cards in a 3-col grid.

const IMG = '../../assets/';

const SOCIALS = [
  { href: 'https://github.com/diegopacheco/',                 title: 'diegopacheco on GitHub',      label: 'GitHub',       img: 'github-logo.png' },
  { href: 'https://br.linkedin.com/in/diegopachecors',        title: 'diegopachecors on LinkedIn',  label: 'LinkedIn',     img: 'linkedin-logo.png' },
  { href: 'http://diego-pacheco.blogspot.com.br/',            title: 'Tech Blog on Blogger',        label: 'Tech Blog',    img: 'Blogger_icon.svg.png' },
  { href: 'https://diegopachecotech.substack.com/',           title: 'diegopachecotech on Substack',label: 'Substack',     img: 'substack_logo.png' },
  { href: 'https://diego-pacheco.medium.com/',                title: 'diego-pacheco on Medium',     label: 'Medium',       img: 'medium.png' },
  { href: 'https://tinyurl.com/diegopacheco',                 title: 'Diego Pacheco YouTube',       label: 'YouTube',      img: 'youtube.png' },
  { href: 'https://twitter.com/diego_pacheco',                title: '@diego_pacheco on Twitter',   label: 'Twitter',      img: 'X_icon.svg.png' },
  { href: 'https://bsky.app/profile/diegopacheco.bsky.social',title: 'diegopacheco.bsky.social',    label: 'Bluesky',      img: 'bluesky.png' },
  { href: 'https://news.ycombinator.com/submitted?id=diegopacheco', title: 'diegopacheco on Hacker News', label: 'Hacker News', img: 'HN.png' },
  { href: 'http://www.slideshare.net/diego.pacheco/presentations',  title: 'diego.pacheco on SlideShare', label: 'SlideShare',  img: 'slideshare.png' },
  { href: 'https://hub.docker.com/u/diegopacheco/',           title: 'diegopacheco on DockerHub',   label: 'DockerHub',    img: 'docker.png' },
  { href: 'https://vimeo.com/channels/pacheco',               title: 'pacheco on Vimeo',            label: 'Vimeo',        img: 'vimeo2.png' },
  { href: 'https://amazon.com/author/diegopacheco',           title: 'Diego Pacheco on Amazon',     label: 'Amazon Author',img: 'amazon-author.png' },
];

function SocialsGrid() {
  return (
    <aside className="socials-column card">
      <h2>Socials</h2>
      <div className="socials-widget">
        <div className="socials-grid">
          {SOCIALS.map((s) => (
            <a
              key={s.label}
              className="social-card"
              href={s.href}
              title={s.title}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={s.label}
            >
              <img src={IMG + s.img} alt={s.label} />
            </a>
          ))}
        </div>
      </div>
    </aside>
  );
}

window.SocialsGrid = SocialsGrid;
