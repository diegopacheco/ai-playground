import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { RouterProvider } from "@tanstack/react-router"
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { router } from "./router"
import "./styles.css"

const client = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } })

createRoot(document.getElementById("root")!).render(<StrictMode><QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider></StrictMode>)
