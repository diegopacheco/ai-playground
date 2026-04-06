import { Outlet } from '@tanstack/react-router'
import { RootLayout } from '../layouts/root-layout'

export default function _root() {
  return (
    <RootLayout>
      <Outlet />
    </RootLayout>
  )
}
