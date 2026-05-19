// LecturesTable.jsx — 🔉 Recent Lectures table, year columns.

function LecturesTable() {
  return (
    <article id="lectures" className="lectures-section">
      <h2>🔉Recent Lectures</h2>
      <table>
        <thead>
          <tr>
            <th>2018</th>
            <th>2016</th>
            <th>2015</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <ul>
                <li><a href="https://www.slideshare.net/diego.pacheco/experiences-building-a-multi-region-cassandra-operations-orchestrator-on-aws" target="_blank" rel="noopener noreferrer">Meetup AWS POA</a></li>
              </ul>
            </td>
            <td>
              <ul>
                <li><a href="https://www.youtube.com/watch?v=Z4_rzsZd70o" target="_blank" rel="noopener noreferrer">Netflix</a></li>
                <li><a href="https://www.infoq.com/br/presentations/microservices-reativos-usando-a-stack-do-netflix-na-aws" target="_blank" rel="noopener noreferrer">QCon SP</a></li>
                <li><a href="http://www.meetup.com/Sao-Paulo-Amazon-Web-Services-AWS-Meetup/events/229283010" target="_blank" rel="noopener noreferrer">AWS SP Meetup</a></li>
              </ul>
            </td>
            <td>
              <ul>
                <li><a href="https://www.infoq.com/br/presentations/vivenciando-devops" target="_blank" rel="noopener noreferrer">QCon SP</a></li>
                <li><a href="https://agilebrazil2015.sched.com/event/4DE2" target="_blank" rel="noopener noreferrer">Agile Brazil</a></li>
              </ul>
            </td>
          </tr>
        </tbody>
      </table>
    </article>
  );
}

window.LecturesTable = LecturesTable;
