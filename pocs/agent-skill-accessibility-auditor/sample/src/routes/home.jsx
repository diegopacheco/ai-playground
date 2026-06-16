const swatch = (a, b) =>
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' width='320' height='200'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='${a}'/><stop offset='1' stop-color='${b}'/></linearGradient></defs><rect width='320' height='200' fill='url(#g)'/></svg>`
  );

const products = [
  { id: 1, name: "Cold Brew Concentrate", price: "$14", img: swatch("#caa472", "#7a5230") },
  { id: 2, name: "Sourdough Starter Kit", price: "$22", img: swatch("#e7d9b0", "#b89b56") },
  { id: 3, name: "Smoked Sea Salt", price: "$9", img: swatch("#d7dbe0", "#8b94a3") },
  { id: 4, name: "Chili Crisp Oil", price: "$12", img: swatch("#e05a3a", "#9c2f17") },
  { id: 5, name: "Matcha Tin", price: "$18", img: swatch("#9bcf8f", "#3f7a44") },
  { id: 6, name: "Wildflower Honey", price: "$16", img: swatch("#f2c14e", "#c08a1e") },
];

const filters = ["Pantry", "Drinks", "On sale"];

export function Home() {
  return (
    <div className="home">
      <section className="hero">
        <h1 id="title">Pantry staples, delivered weekly</h1>
        <h3 className="muted">Small-batch goods from independent makers.</h3>
        <div className="searchrow">
          <input className="search" type="search" placeholder="Search products" />
          <div className="chips">
            {filters.map((f) => (
              <div className="chip" role="button" key={f}>
                {f}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="grid">
        {products.map((p) => (
          <article className="prod" key={p.id}>
            <img src={p.img} />
            <div className="prodbody">
              <div className="pname">{p.name}</div>
              <div className="price">
                {p.price} <span className="deal">save 20%</span>
              </div>
            </div>
          </article>
        ))}
      </section>

      <section className="news">
        <h3 id="title">Get the weekly drop</h3>
        <p className="muted">One email, every Friday. No spam.</p>
        <form className="newsform" onSubmit={(e) => e.preventDefault()}>
          <input className="email" type="email" placeholder="you@email.com" tabIndex={5} />
          <button className="subscribe" type="submit">
            Subscribe
          </button>
        </form>
      </section>
    </div>
  );
}
