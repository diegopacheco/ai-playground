// BooksCarousel.jsx — overlapping horizontal carousel of book covers.
// Prev / Next buttons; book cards overlap by 112 px.

const IMG = '../../assets/';

const BOOKS = [
  { href: 'https://diegopacheco.github.io/The-Art-of-Sense-A-Philosophy-of-Modern-AI/',          img: `${IMG}book-cover-AOS-2025-AI.png`,        alt: "The Art of Sense: A Philosophy of Modern AI" },
  { href: 'https://diegopacheco.github.io/diegopacheco-architecture-library/introduction.html',  img: `${IMG}book-4-2025-cover.png`,             alt: "Software Architecture Library" },
  { href: 'https://www.amazon.com/Continuous-Modernization-never-ending-microservices-distributed-ebook/dp/B0DHS63NQH/', img: `${IMG}book-cm-2024.png`, alt: "Continuous Modernization" },
  { href: 'https://bpbonline.com/products/principles-of-software-architecture-modernization',    img: `${IMG}book-arch-monoliths-2023.png`,      alt: "Principles of Software Architecture Modernization" },
  { href: 'https://books.apple.com/us/book/building-applications-with-scala/id1113861297',       img: `${IMG}book-cover-scala-2016-2.png`,       alt: "Building Applications with Scala" },
  { href: 'https://www.packtpub.com/application-development/building-effective-microservices-video', img: `${IMG}video-cover-microservices-2017.png`, alt: "Building Effective Microservices Video" },
];

function BooksCarousel() {
  const sliderRef = React.useRef(null);
  const [index, setIndex] = React.useState(0);
  const OVERLAP = 140;

  const step = React.useCallback(() => {
    if (!sliderRef.current) return 0;
    const card = sliderRef.current.querySelector('.book-card');
    if (!card) return 0;
    return card.getBoundingClientRect().width - OVERLAP;
  }, []);

  React.useEffect(() => {
    if (!sliderRef.current) return;
    sliderRef.current.style.transform = `translateX(${-index * step()}px)`;
  }, [index, step]);

  const scroll = (delta) => {
    setIndex((prev) => Math.min(BOOKS.length - 1, Math.max(0, prev + delta)));
  };

  return (
    <div className="published-works">
      <h2>Published Books & Video Series</h2>
      <div className="books-carousel">
        <div className="books-slider" id="books-slider" ref={sliderRef}>
          {BOOKS.map((b, i) => (
            <a
              key={i}
              className="book-card"
              href={b.href}
              target="_blank"
              rel="noopener noreferrer"
            >
              <img src={b.img} alt={b.alt} />
            </a>
          ))}
        </div>
        <div className="carousel-controls">
          <button id="prev-book" onClick={() => scroll(-1)}>Prev</button>
          <button id="next-book" onClick={() => scroll(1)}>Next</button>
        </div>
      </div>
    </div>
  );
}

window.BooksCarousel = BooksCarousel;
