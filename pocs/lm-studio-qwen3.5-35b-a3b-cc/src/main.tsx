import { StrictMode } from 'react'
import { render } from 'react-dom'
import { RouterProvider } from '@tanstack/react-router'
import { router } from './router'

render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
  document.getElementById('root')!
)
