import { Outlet } from '@tanstack/react-router'
import { RootLayout } from '../layouts/root-layout'

export const Route = {
  component: () => (
    <RootLayout>
      <Outlet />
    </RootLayout>
  ),
}
