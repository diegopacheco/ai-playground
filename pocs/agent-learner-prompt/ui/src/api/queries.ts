import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import * as api from './client'
import type { TaskRequest, ConfigRequest } from '../types'

export function useProjects() {
  return useQuery({
    queryKey: ['projects'],
    queryFn: api.getProjects,
  })
}

export function useProject(name: string | null) {
  return useQuery({
    queryKey: ['project', name],
    queryFn: () => api.getProject(name!),
    enabled: !!name,
  })
}

export function useTaskStatus(taskId: string | null) {
  return useQuery({
    queryKey: ['task', taskId],
    queryFn: () => api.getTaskStatus(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      const data = query.state.data
      return data?.completed ? false : 2000
    },
  })
}

export function useConfig() {
  return useQuery({
    queryKey: ['config'],
    queryFn: api.getConfig,
  })
}

export function useSubmitTask() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (req: TaskRequest) => api.submitTask(req),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
    },
  })
}

export function useUpdateConfig() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (req: ConfigRequest) => api.updateConfig(req),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] })
    },
  })
}
