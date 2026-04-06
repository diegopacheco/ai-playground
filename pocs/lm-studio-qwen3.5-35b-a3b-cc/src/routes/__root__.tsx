import { Outlet, createFileRoute } from '@tanstack/react-router'
import { RootLayout } from '../layouts/root-layout'

export const route = createFileRoute('/')({
  component: _Root_,
})

function _Root_() {
  return (
    <RootLayout>
      <Outlet />
    </RootLayout>
  )
}
