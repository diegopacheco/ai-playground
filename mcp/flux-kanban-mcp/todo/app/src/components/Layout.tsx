import { Outlet } from '@tanstack/react-router'
import { ListSidebar } from './ListSidebar'

export function Layout() {
  return (
    <div className="app-layout">
      <ListSidebar />
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  )
}
