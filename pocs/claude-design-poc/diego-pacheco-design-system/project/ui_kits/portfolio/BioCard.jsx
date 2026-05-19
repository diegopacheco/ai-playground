// BioCard.jsx — 👨‍💻 long-form bio + 🌱 Currently paragraph.

function BioCard() {
  return (
    <section className="bio-section card">
      <h3>👨‍💻 Diego Pacheco Bio</h3>
      <div className="bio-text">
        <p>
          Diego Pacheco is a seasoned, experienced 🇧🇷Brazilian software architect,
          author, speaker, technology mentor, and DevOps practitioner with more
          than 20+ years of solid experience. I've been building teams and
          mentoring people for more than a decade, teaching soft skills and
          technology daily. Selling projects, hiring, building solutions, running
          coding dojos, long retrospectives, weekly 1:1s, design sessions, code
          reviews, and my favorite debate club: architects community of practices
          and development groups for more than a decade. Live, breathe, and
          practice real Agile since 2005, coaching teams have helped many
          companies to discover better ways to work using Lean and Kanban, Agile
          principles, and methods like XP and DTA/TTA. I've led complex
          architecture teams and engineering teams at scale guided by SOA
          principles, using a variety of open-source languages like Java, Scala,
          Rust, Go, Python, Groovy, JavaScript and TypeScript, cloud providers
          like AWS Cloud and Google GCP, amazing solutions like Akka, ActiveMQ,
          Netty, Tomcat and Gatling, NoSQL databases like Cassandra, Redis,
          Elasticache Redis, Elasticsearch, Opensearch, RabbitMQ, libraries like
          Spring, Hibernate, and Spring Boot and also the NetflixOSS Stack:
          Simian Army, RxJava, Karyon, Dynomite, Eureka, and Ribbon. I've
          implemented complex security solutions at scale using AWS KMS, S3,
          Containers (ECS and EKs), Terraform, and Jenkins. Over a decade of
          experience as a consultant, coding, designing, and training people at
          big customers in Brazil, London, Barcelona, India, and the
          USA(Silicon Valley and Midwest). I have a passion for functional
          programming and distributed systems, NoSQL Databases, a mindset for
          Observability, and always learning new programming languages.
        </p>
      </div>
      <div className="currently-text">
        <p>
          🌱Currently: Working as a principal Software Architect with AWS public
          cloud, Kubernetes/EKS, performing complex cloud migrations, library
          migrations, server and persistence migrations, and security at scale
          with multi-level envelope encryption solutions using KMS and S3. While
          still hiring, teaching, mentoring, and growing engineers and
          architects. During my free time, I love playing guitar, gaming, coding
          POCs, and blogging. Active blogger blog at{' '}
          <a href="http://diego-pacheco.blogspot.com.br/" target="_blank" rel="noopener noreferrer">
            http://diego-pacheco.blogspot.com.br/
          </a>
        </p>
      </div>
    </section>
  );
}

window.BioCard = BioCard;
