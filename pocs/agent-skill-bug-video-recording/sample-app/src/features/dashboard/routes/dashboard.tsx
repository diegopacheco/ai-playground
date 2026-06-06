import { useQuery } from '@tanstack/react-query'
import { fetchDashboard } from '../api/dashboard'
import { Stats } from '../components/Stats'

export function DashboardPage() {
  const { data, isPending } = useQuery({ queryKey: ['dashboard'], queryFn: fetchDashboard })
  if (isPending || !data) return <p className="page">Loading…</p>
  return (
    <section className="page">
      <h1>Dashboard</h1>
      <Stats metrics={data.metrics!} />
    </section>
  )
}
