// Hero.jsx — top card: gradient bg, profile photo + tagline + roles list,
// plus the mini-essays slider, side-project stack, and books carousel.

const IMG = '../../assets/'; // path from index.html

function ProfileImage() {
  return (
    <div className="profile-image-container">
      <img src={`${IMG}image-profile.png`} alt="Diego Pacheco Profile" />
    </div>
  );
}

function HeroHeading() {
  return (
    <div className="hero-heading">
      <div className="hero-title-row">
        <h1>Diego Pacheco</h1>
        <span className="hero-about">About me</span>
      </div>
      <ul className="hero-roles">
        <li>Father</li>
        <li>Cat's Father</li>
        <li>Principal Software Architect</li>
        <li>SOA Expert</li>
        <li>DevOps Practitioner</li>
        <li>Author</li>
        <li>Mentor</li>
        <li>Speaker</li>
        <li>Leader</li>
      </ul>
    </div>
  );
}

function Hero() {
  return (
    <header className="hero card">
      <div className="hero-left">
        <p className="eyebrow">
          Principal Software Architect • Author • Speaker • Mentor • Leader
        </p>
        <div className="hero-header">
          <ProfileImage />
          <HeroHeading />
        </div>
        <p className="hero-description">
          Brazilian software architect, SOA expert, DevOps practitioner, author,
          mentor, and speaker.
        </p>
      </div>

      <div className="hero-right">
        <div className="hero-right-main">
          <div
            className="mini-carousel-and-stack"
            style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}
          >
            <MiniCardGrid />
            <SideProjectStack />
          </div>
          <BooksCarousel />
        </div>
      </div>
    </header>
  );
}

window.Hero = Hero;
