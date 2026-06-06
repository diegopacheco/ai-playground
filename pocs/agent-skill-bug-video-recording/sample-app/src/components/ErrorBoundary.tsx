import { Component } from 'react'
import type { ReactNode } from 'react'

type Props = { children: ReactNode }
type State = { error: Error | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="crash" role="alert">
          <h2>Something went wrong</h2>
          <p className="crash-message">{this.state.error.message}</p>
        </div>
      )
    }
    return this.props.children
  }
}
