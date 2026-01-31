import { useQuery } from '@tanstack/react-query';
import { getAgents, getDebate, getDebates } from '../api/debates';

export function useAgents() {
  return useQuery({
    queryKey: ['agents'],
    queryFn: getAgents,
  });
}

export function useDebates() {
  return useQuery({
    queryKey: ['debates'],
    queryFn: getDebates,
  });
}

export function useDebate(id: string | null) {
  return useQuery({
    queryKey: ['debate', id],
    queryFn: () => (id ? getDebate(id) : Promise.resolve(null)),
    enabled: !!id,
  });
}
