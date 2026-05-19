// MiniCardGrid.jsx — the 📝 Tiny Essays grid of language-emoji mini-cards.
// Tech-language icons come from CDNs (slackmojis / vlang.io), matching the
// live site exactly.

const ESSAYS = [
  {
    name: 'Rust',
    href: 'https://gist.github.com/diegopacheco/4b7dfeb781ad3455ae2a6b090d9deaa7',
    img: 'https://emojis.slackmojis.com/emojis/images/1643514229/1965/rust.png?1643514229',
  },
  {
    name: 'Scala',
    href: 'https://gist.github.com/diegopacheco/1b5df4287dd1ce4276631fd630267311',
    img: 'https://slackmojis.com/emojis/1857-scala/download',
  },
  {
    name: 'Zig',
    href: 'https://gist.github.com/diegopacheco/7d7c8110db68352d58a18b0e3e3c2bb0',
    img: 'https://slackmojis.com/emojis/57099-ziglang/download',
  },
  {
    name: 'Kotlin',
    href: 'https://gist.github.com/diegopacheco/f6beabf1451cfe1ec2dc89a19a78fdc5',
    img: 'https://slackmojis.com/emojis/2351-kotlin/download',
  },
  {
    name: 'Clojure',
    href: 'https://gist.github.com/diegopacheco/9453877378f007e8903a359f298a0afa',
    img: 'https://slackmojis.com/emojis/378-clojure/download',
  },
  {
    name: 'Haskell',
    href: 'https://gist.github.com/diegopacheco/057087dc7ae236bdd0700014a31c88ef',
    img: 'https://slackmojis.com/emojis/22333-haskell/download',
  },
  {
    name: 'Nim',
    href: 'https://gist.github.com/diegopacheco/0fb84d881e2423147d9cb6f8619bf473',
    img: 'https://slackmojis.com/emojis/63427-nim/download',
  },
  {
    name: 'V',
    href: 'https://gist.github.com/diegopacheco/3d0b176eb83e569da582a0770209e22f',
    img: 'https://vlang.io/img/v-logo.png',
  },
];

function MiniCard({ name, href, img }) {
  return (
    <a
      className="mini-card"
      href={href}
      title={`${name} essay`}
      target="_blank"
      rel="noopener noreferrer"
    >
      <img src={img} alt={`${name} icon`} />
      <span className="mini-label">{name}</span>
    </a>
  );
}

function MiniCardGrid() {
  return (
    <div className="mini-carousel">
      <h3>📝 Tiny Essays</h3>
      <div className="books-slider mini-slider" id="essays-slider">
        {ESSAYS.map((e) => (
          <MiniCard key={e.name} {...e} />
        ))}
      </div>
    </div>
  );
}

window.MiniCard = MiniCard;
window.MiniCardGrid = MiniCardGrid;
