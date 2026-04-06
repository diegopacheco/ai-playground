import { Outlet, createFileRoute } from '@tanstack/react-router'
import { RootLayout } from '../layouts/root-layout'

export default function _root() {
  return (
    <RootLayout>
      <Outlet />
    </RootLayout>
  )
}

export const route = createFileRoute('/$')({
  component: _root,
})
