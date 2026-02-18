import { createContext, useContext, useState, ReactNode } from 'react'
import { AuthUser } from './types'

interface AuthContextType {
  user: AuthUser | null
  login: (user: AuthUser) => void
  logout: () => void
}

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() => {
    const stored = localStorage.getItem('auth')
    return stored ? JSON.parse(stored) : null
  })

  const login = (u: AuthUser) => {
    setUser(u)
    localStorage.setItem('auth', JSON.stringify(u))
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem('auth')
  }

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
