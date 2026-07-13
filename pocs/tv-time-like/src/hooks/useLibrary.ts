import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type { Media } from "../../shared/types"
import { api } from "../api/client"

export const useLibrary = () => useQuery({ queryKey: ["library"], queryFn: api.library })

export const useLibraryActions = () => {
  const client = useQueryClient()
  const refresh = () => client.invalidateQueries({ queryKey: ["library"] })
  const refreshDiscover = () => {
    client.invalidateQueries({ queryKey: ["library"] })
    client.invalidateQueries({ queryKey: ["search"] })
  }
  return {
    add: useMutation({ mutationFn: (media: Media) => api.add(media), onSuccess: refreshDiscover }),
    remove: useMutation({ mutationFn: api.remove, onSuccess: refresh }),
    toggleMedia: useMutation({ mutationFn: api.toggleMedia, onSuccess: refresh }),
    toggleEpisode: useMutation({ mutationFn: api.toggleEpisode, onSuccess: refresh }),
    importData: useMutation({ mutationFn: api.importTvTime, onSuccess: refresh }),
    syncMetadata: useMutation({ mutationFn: api.syncMetadata, onSuccess: refresh }),
    syncEpisodes: useMutation({ mutationFn: api.syncEpisodes, onSuccess: refresh }),
    syncMediaEpisodes: useMutation({ mutationFn: api.syncMediaEpisodes, onSuccess: refresh })
  }
}
