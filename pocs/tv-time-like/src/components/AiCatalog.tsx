import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import type { AiProvider, Media } from "../../shared/types"
import { api } from "../api/client"
import { Icon } from "./Icon"
import { Poster } from "./Poster"

const providers: { id: AiProvider; label: string }[] = [{ id: "codex", label: "Codex CLI" }, { id: "claude", label: "Claude CLI" }, { id: "gemini", label: "Gemini CLI" }]

export function AiCatalog({ onAdd }: { onAdd: (media: Media) => void }) {
  const client = useQueryClient()
  const [topic, setTopic] = useState("")
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings })
  const save = useMutation({ mutationFn: api.saveSettings, onSuccess: data => client.setQueryData(["settings"], data) })
  const catalog = useMutation({ mutationFn: ({ refresh }: { refresh: boolean }) => api.aiCatalog(topic, refresh) })
  return <section className="ai-section">
    <div className="ai-heading"><div><span className="eyebrow">Agent-curated catalog</span><h2>Let an AI scout the web.</h2><p>Fresh finds and older essentials, researched live and cached for six hours.</p></div><div className="provider-picker"><label htmlFor="ai-provider">Research with</label><select id="ai-provider" value={settings.data?.aiProvider || "codex"} onChange={event => save.mutate({ aiProvider: event.target.value as AiProvider })}>{providers.map(provider => <option value={provider.id} key={provider.id}>{provider.label}</option>)}</select></div></div>
    <div className="ai-search"><input value={topic} onChange={event => setTopic(event.target.value)} placeholder="A mood, genre, era or anything at all…"/><button className="button ink" onClick={() => catalog.mutate({ refresh: false })} disabled={catalog.isPending}><Icon name="search" size={17}/>{catalog.isPending ? "Researching the web…" : "Build catalog"}</button></div>
    {catalog.error && <div className="ai-error"><strong>The research run did not finish.</strong><span>{catalog.error.message}</span></div>}
    {catalog.data && <><div className="catalog-meta"><span>{catalog.data.items.length} researched titles · {catalog.data.provider}{catalog.data.cached ? " · cached" : " · just updated"}</span><button onClick={() => catalog.mutate({ refresh: true })}>Refresh research <Icon name="arrow" size={15}/></button></div><div className="catalog-strip">{catalog.data.items.map(item => <article className="catalog-card" key={item.media.id}><Poster media={item.media}/><div><span className="media-kicker">{item.media.type} · {item.media.year}</span><h3>{item.media.title}</h3><p>{item.reason}</p><div className="catalog-actions"><button onClick={() => onAdd(item.media)}><Icon name="plus" size={15}/>Track</button><a href={item.sourceUrl} target="_blank" rel="noreferrer">{item.sourceName}<Icon name="arrow" size={14}/></a></div></div></article>)}</div></>}
  </section>
}
