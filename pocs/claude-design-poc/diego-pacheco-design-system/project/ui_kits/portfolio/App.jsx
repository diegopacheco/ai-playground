// App.jsx — composes the full portfolio page.

const SKILLS = [
  { label: 'Architecture Design and architecture coding for highly scalable systems' },
  { label: 'Delivering distributed systems using SOA and Microservices principles, tools, and techniques' },
  { label: 'Driving and executing complex cloud migrations, library and server migrations at scale' },
  { label: 'Performance tuning, troubleshooting & DevOps engineering' },
  { label: 'Functional Programming and Scala' },
  { label: 'Technology Mentor, agile coach & leader for architecture and engineering teams' },
  { label: 'Consultant on development practices with XP / Kanban' },
  { label: 'Hire, develop, retain, and truly grow talent at scale' },
];

const POCS = [
  { label: 'AI Playground',               href: 'https://github.com/diegopacheco/ai-playground' },
  { label: 'Data Structures',             href: 'https://github.com/diegopacheco/data-structures' },
  { label: 'Servers Benchmark',           href: 'https://github.com/diegopacheco/servers-benchmark' },
  { label: 'IF Alternatives',             href: 'https://github.com/diegopacheco/java-pocs/tree/master/pocs/if-alternatives-fun' },
  { label: 'IF Killer Proper OOP',        href: 'https://github.com/diegopacheco/java-pocs/tree/master/pocs/if-killer-proper-oop' },
  { label: 'Spring Boot Virtual Threads', href: 'https://github.com/diegopacheco/java-pocs/tree/master/pocs/java-21-spring-boot-3-async-virtual-threads' },
  { label: 'Netty K6',                    href: 'https://github.com/diegopacheco/scala-playground/tree/master/scala-3.7-spring-boot-3.5-netty-multi-workers-metrics-grafana-k6' },
  { label: 'Scala 3x Patterns',           href: 'https://github.com/diegopacheco/scala-playground' },
  { label: 'Postgres Partition Playground', href: 'https://github.com/diegopacheco/devops-playground/tree/master/postgres-partitions-playground' },
  { label: 'OOP Anti-Patterns',           href: 'https://github.com/diegopacheco/java-pocs/tree/master/pocs/oop-anti-patterns' },
];

const PAPERS = [
  { label: 'Arch, Business & Value',   href: 'https://ilegra.com/wp-content/uploads/2019/10/Arquitetura-Valor-e-Negócio.pdf' },
  { label: 'Multi/Poly Cloud',         href: 'https://ilegra.com/wp-content/uploads/2019/10/Paper-Multicloud.pdf' },
  { label: 'Micro Frontends & SRR',    href: 'https://ilegra.com/wp-content/uploads/2019/10/Paper-Frontend.pdf' },
];

const CERTS = [
  { label: '📐Google Cloud Architect' },
  { label: '☕ Sun Certified Web Component Developer' },
  { label: '☕ Sun Certified Java Programmer' },
];

const AI_POCS = [
  { label: '🏙️ Code City Viz',          href: 'https://github.com/diegopacheco/code-city-viz' },
  { label: '📄 Rust Arxiv Sumarizer',  href: 'https://github.com/diegopacheco/ras' },
  { label: '🦀 Claudio Coda',          href: 'https://github.com/diegopacheco/claudio-coda-rust' },
  { label: '🔮 Local Agent Orama',     href: 'https://github.com/diegopacheco/local-agent-orama' },
  { label: '🦙 Local Agent llr3',      href: 'https://github.com/diegopacheco/local-agent-rust-llama3' },
  { label: '🌐 Multi-Agent-Verse',     href: 'https://github.com/diegopacheco/multi-agent-verse' },
  { label: '☸️ prompt-2-k8s-agent',    href: 'https://github.com/diegopacheco/prompt-2-k8s-agent' },
  { label: '📚 Agent Learner Prompt',  href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-learner-prompt' },
  { label: '📊 Prompt Score',          href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/prompt-score' },
  { label: '🤖 Self Training',         href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/self-training-machine' },
  { label: '🎮 Connect Four Agents',   href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/connect-four-agent-vs-agent' },
  { label: '💬 Agent Debate Club',     href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-debate-club' },
  { label: '🏛️ Agents Auction House',  href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/agents-auction-hourse' },
  { label: '🎲 AI RPG',                href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/ai-rpg' },
  { label: '📋 Skill Evaluator',       href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/skill-evaluator' },
  { label: '🔐 Leak Detector Skill',   href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/leak-detector-skill' },
  { label: '☸️ K8s SRE Agent',         href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/k8s-sre-agent-operator' },
  { label: '⚡ AutoBench Skill',       href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/autobench-skill-poc/sample-bench' },
  { label: '🏛️ Multi-Agent Auction',   href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/agents-auction-hourse' },
  { label: '🐺 Werewolf Agent Game',   href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-werewolf' },
  { label: '📖 Runbook Generator',     href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-runbook' },
  { label: '⏱️ Tool Time Tracker',     href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/cc-hook-tool-time-tracker' },
  { label: '🏗️ Infra Automation Gen',  href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-skill-infra-automation-generator' },
  { label: '🔧 Bruno Collection Gen',  href: 'https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-bruno-skill' },
];

function App() {
  return (
    <div className="site-shell">
      <Hero />
      <BioCard />
      <div className="main-content-area">
        <SocialsGrid />
        <section className="details-column card">
          <div className="skills-lectures-container">
            <div className="skills-column">
              <ListCard
                id="skills"
                heading="💻 Core skills and expertise:"
                items={SKILLS}
                listClass="skills-list"
              />
              <ListCard
                id="feature-pocs"
                heading="🧪 Feature POCs:"
                items={POCS}
                listClass="feature-pocs-list"
              />
            </div>
            <div className="lectures-column">
              <LecturesTable />
              <ListCard
                id="papers"
                heading="📜 Papers (2019)"
                headingLevel="h2"
                items={PAPERS}
                listClass="papers-list"
              />
              <ListCard
                id="certifications"
                heading="💯 Certifications"
                items={CERTS}
                listClass="certifications-list"
              />
            </div>
          </div>
          <ListCard
            id="feature-ai-pocs"
            heading="🤖 Feature AI POCs:"
            items={AI_POCS}
            listClass="feature-ai-pocs-list"
          />
        </section>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
