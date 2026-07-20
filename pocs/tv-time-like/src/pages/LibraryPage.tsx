import { useMemo, useState } from "react"
import { Icon } from "../components/Icon"
import { ImportDialog } from "../components/ImportDialog"
import { LibraryGridCard } from "../components/LibraryGridCard"
import { LibraryDetailDialog } from "../components/LibraryDetailDialog"
import { useLibrary, useLibraryActions } from "../hooks/useLibrary"

type Filter = "all" | "movie" | "show" | "wip"
type Sort = "latest" | "alpha" | "alphaDesc" | "newest" | "oldest" | "release" | "progress"

const progressFor = (item: import("../../shared/types").Media) => item.type === "movie" ? Number(item.watched) * 100 : Math.round(item.episodes.filter(episode => episode.watched).length / Math.max(item.episodes.length, 1) * 100)

const latestFor = (item: import("../../shared/types").Media) => item.type === "movie" ? String(item.addedAt || "") : item.episodes.reduce((last, episode) => episode.watched && episode.watchedAt && episode.watchedAt > last ? episode.watchedAt : last, "") || String(item.addedAt || "")

export function LibraryPage() {
  const library = useLibrary()
  const actions = useLibraryActions()
  const [filter, setFilter] = useState<Filter>("all")
  const [query, setQuery] = useState("")
  const [sort, setSort] = useState<Sort>("latest")
  const [page, setPage] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [importOpen, setImportOpen] = useState(false)
  const media = useMemo(() => {
    const items = (library.data?.media || []).filter(item => {
      const matchesFilter = filter === "all" ? true : filter === "wip" ? item.type === "show" && progressFor(item) < 100 : item.type === filter
      return matchesFilter && item.title.toLowerCase().includes(query.toLowerCase())
    })
    return items.toSorted((a, b) => {
      if (sort === "latest") return latestFor(b).localeCompare(latestFor(a)) || a.title.localeCompare(b.title)
      if (sort === "alpha") return a.title.localeCompare(b.title)
      if (sort === "alphaDesc") return b.title.localeCompare(a.title)
      if (sort === "newest") return String(b.addedAt).localeCompare(String(a.addedAt))
      if (sort === "oldest") return String(a.addedAt).localeCompare(String(b.addedAt))
      if (sort === "release") return b.year - a.year || a.title.localeCompare(b.title)
      return progressFor(b) - progressFor(a) || a.title.localeCompare(b.title)
    })
  }, [library.data, filter, query, sort])
  const pageSize = 24
  const pageCount = Math.max(1, Math.ceil(media.length / pageSize))
  const visible = media.slice(page * pageSize, page * pageSize + pageSize)
  const busy = actions.remove.isPending || actions.toggleMedia.isPending || actions.toggleEpisode.isPending
  return <div className="page library-page">
    <section className="page-heading"><div><span className="eyebrow">Collected & remembered</span><h1>My library</h1><p>Every title you are watching, finished or saving for later.</p></div><div className="heading-actions"><button className="button secondary import-button" onClick={() => actions.syncEpisodes.mutate()} disabled={actions.syncEpisodes.isPending}><Icon name="tv" size={17}/>{actions.syncEpisodes.isPending ? "Syncing episodes…" : "Sync episodes"}</button><button className="button secondary import-button" onClick={() => actions.syncMetadata.mutate()} disabled={actions.syncMetadata.isPending}><Icon name="film" size={17}/>{actions.syncMetadata.isPending ? "Loading artwork…" : "Refresh artwork"}</button><button className="button secondary import-button" onClick={() => setImportOpen(true)}><Icon name="upload" size={17}/>Import TV Time</button></div></section>
    <section className="library-toolbar"><div className="segmented">{(["all", "movie", "show", "wip"] as Filter[]).map(value => <button key={value} className={filter === value ? "active" : ""} onClick={() => { setFilter(value); setPage(0) }}>{value === "all" ? "Everything" : value === "movie" ? "Movies" : value === "show" ? "TV shows" : "W.I.P"}</button>)}</div><label className="library-search"><Icon name="search" size={16}/><input value={query} onChange={event => { setQuery(event.target.value); setPage(0) }} placeholder="Filter your library"/></label><label className="sort-select"><span>Sort</span><select value={sort} onChange={event => { setSort(event.target.value as Sort); setPage(0) }}><option value="latest">Latest</option><option value="alpha">Alphabetical</option><option value="alphaDesc">Reverse alphabetical</option><option value="newest">Recently added</option><option value="oldest">Oldest added</option><option value="release">Release date</option><option value="progress">Highest progress</option></select></label><span>{media.length} titles</span></section>
    {library.isLoading ? <div className="library-loading"/> : media.length ? <><div className="library-cover-grid">{visible.map(item => <LibraryGridCard key={item.id} media={item} onOpen={() => setSelectedId(item.id)}/>)}</div><div className="pagination"><button disabled={page === 0} onClick={() => setPage(value => value - 1)}>Previous</button><span>Page {page + 1} of {pageCount}</span><button disabled={page + 1 >= pageCount} onClick={() => setPage(value => value + 1)}>Next</button></div></> : <div className="empty-library"><div className="empty-reels"><i/><i/><i/></div><span className="eyebrow">Nothing filed here yet</span><h2>Your next favorite belongs here.</h2><p>Add a title from Discover or bring over your TV Time history.</p><div><a href="/" className="button primary"><Icon name="search" size={17}/>Find a title</a><button className="button secondary" onClick={() => setImportOpen(true)}><Icon name="upload" size={17}/>Import history</button></div></div>}
    <ImportDialog open={importOpen} onClose={() => setImportOpen(false)} busy={actions.importData.isPending} onImport={async text => actions.importData.mutateAsync(text)}/>
    <LibraryDetailDialog media={library.data?.media.find(item => item.id === selectedId) || null} onClose={() => setSelectedId(null)} onMediaToggle={id => actions.toggleMedia.mutate(id)} onEpisodeToggle={id => actions.toggleEpisode.mutate(id)} onRemove={id => actions.remove.mutate(id)} busy={busy}/>
  </div>
}
