import { useQuery } from '@tanstack/react-query'
import { fetchHistory } from '../api/client'

export function useMatchHistory() {
  return useQuery({
    queryKey: ['history'],
    queryFn: fetchHistory,
    refetchInterval: 5000,
  })
}
