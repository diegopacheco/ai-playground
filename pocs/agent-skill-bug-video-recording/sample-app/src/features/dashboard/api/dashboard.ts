export type Stat = { id: string; label: string; value: number }

export function fetchDashboard(): Promise<{ stats: Stat[]; metrics?: Stat[] }> {
  const stats: Stat[] = [
    { id: 's1', label: 'Orders', value: 128 },
    { id: 's2', label: 'Revenue', value: 5400 },
    { id: 's3', label: 'Visitors', value: 2310 }
  ]
  return new Promise(resolve => setTimeout(() => resolve({ stats }), 150))
}
