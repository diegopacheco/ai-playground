import { MetricCard } from "../components/MetricCard"
import { ProgressRing } from "../components/ProgressRing"
import { useLibrary } from "../hooks/useLibrary"

const hours = (minutes: number) => `${Math.floor(minutes / 60)}h ${minutes % 60}m`
const calendarTime = (minutes: number) => {
  const days = Math.floor(minutes / 1_440)
  const months = Math.floor(days / 30)
  return `${months} months · ${days % 30} days · ${Math.floor(minutes % 1_440 / 60)} hours`
}

export function MetricsPage() {
  const library = useLibrary()
  const metrics = library.data?.metrics
  if (!metrics) return <div className="page"><div className="library-loading"/></div>
  const maxMonth = Math.max(...metrics.monthlyActivity.map(item => item.count), 1)
  const maxGenre = Math.max(...metrics.genreBreakdown.map(item => item.count), 1)
  return <div className="page metrics-page">
    <section className="page-heading metrics-heading"><div><span className="eyebrow">The story in numbers</span><h1>Your viewing,<br/><em>by the numbers.</em></h1></div><div className="time-total"><small>Total screen time</small><strong>{hours(metrics.totalMinutes)}</strong><span>{calendarTime(metrics.totalMinutes)}</span></div></section>
    <section className="metric-grid">
      <MetricCard label="Movies watched" value={String(metrics.movieCount).padStart(2, "0")} detail={calendarTime(metrics.movieMinutes)} icon="film"/>
      <MetricCard label="Shows followed" value={String(metrics.showCount).padStart(2, "0")} detail={`${metrics.episodeCount} episodes`} icon="tv"/>
      <MetricCard label="Episodes watched" value={String(metrics.episodeCount).padStart(2, "0")} detail={calendarTime(metrics.showMinutes)} icon="play"/>
      <MetricCard label="Average a week" value={hours(metrics.averageWeeklyMinutes)} detail={metrics.trackingSince ? `Since ${new Date(metrics.trackingSince).getFullYear()}` : "No watch history yet"} icon="clock"/>
    </section>
    <section className="analytics-grid">
      <article className="analytics-card activity-card"><div className="card-heading"><div><span className="eyebrow">Viewing rhythm</span><h2>Months in motion</h2></div><span>Titles + episodes</span></div><div className="activity-chart">{metrics.monthlyActivity.map(item => <div key={item.month} className="month-column"><div className="bar-track"><i style={{ height: `${Math.max(item.count / maxMonth * 100, item.count ? 8 : 2)}%` }}/></div><span>{item.month}</span></div>)}</div></article>
      <article className="analytics-card completion-card"><span className="eyebrow">Series progress</span><h2>Completion rate</h2><ProgressRing value={metrics.completionRate} label="episodes"/><p>{metrics.episodeCount ? "You are steadily clearing the queue." : "Mark episodes watched to chart your progress."}</p></article>
      <article className="analytics-card genre-card"><div className="card-heading"><div><span className="eyebrow">Taste profile</span><h2>Genres you return to</h2></div><span>{metrics.genreBreakdown.length} genres</span></div>{metrics.genreBreakdown.length ? <div className="genre-chart">{metrics.genreBreakdown.slice(0, 6).map((item, index) => <div className="genre-line" key={item.genre}><span className="genre-rank">{String(index + 1).padStart(2, "0")}</span><strong>{item.genre}</strong><div><i style={{ width: `${item.count / maxGenre * 100}%` }}/></div><b>{item.count}</b></div>)}</div> : <div className="chart-empty">Your genre profile will appear as the library grows.</div>}</article>
    </section>
  </div>
}
