import { useQuery } from '@tanstack/react-query'
import { fetchAgents, fetchAgent } from '../api/agents'

export function useAgents() {
  return useQuery({
    queryKey: ['agents'],
    queryFn: fetchAgents,
    refetchInterval: 3000,
  })
}

export function useAgent(id: string | null) {
  return useQuery({
    queryKey: ['agent', id],
    queryFn: () => fetchAgent(id!),
    enabled: !!id,
  })
}
