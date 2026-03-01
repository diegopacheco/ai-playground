import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import Card from "./Card"
import type { CardData } from "../types"

describe("Card", () => {
  const baseCard: CardData = {
    position: 0,
    value: 1,
    flipped: false,
    matched: false,
  }

  it("renders question mark when face down", () => {
    render(<Card card={baseCard} index={0} onClick={vi.fn()} disabled={false} />)
    expect(screen.getByText("?")).toBeInTheDocument()
  })

  it("renders card value when flipped", () => {
    const flippedCard: CardData = { ...baseCard, flipped: true }
    render(<Card card={flippedCard} index={0} onClick={vi.fn()} disabled={false} />)
    expect(screen.getByText("1")).toBeInTheDocument()
  })

  it("renders card value when matched", () => {
    const matchedCard: CardData = { ...baseCard, matched: true }
    render(<Card card={matchedCard} index={0} onClick={vi.fn()} disabled={false} />)
    expect(screen.getByText("1")).toBeInTheDocument()
  })

  it("calls onClick when face down card is clicked", () => {
    const onClick = vi.fn()
    render(<Card card={baseCard} index={0} onClick={onClick} disabled={false} />)
    fireEvent.click(screen.getByText("?").closest("div")!)
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it("does not call onClick when card is already flipped", () => {
    const onClick = vi.fn()
    const flippedCard: CardData = { ...baseCard, flipped: true }
    render(<Card card={flippedCard} index={0} onClick={onClick} disabled={false} />)
    fireEvent.click(screen.getByText("1").closest("div")!)
    expect(onClick).not.toHaveBeenCalled()
  })

  it("does not call onClick when disabled", () => {
    const onClick = vi.fn()
    render(<Card card={baseCard} index={0} onClick={onClick} disabled={true} />)
    fireEvent.click(screen.getByText("?").closest("div")!)
    expect(onClick).not.toHaveBeenCalled()
  })

  it("applies matched styling when card is matched", () => {
    const matchedCard: CardData = { ...baseCard, matched: true }
    const { container } = render(<Card card={matchedCard} index={0} onClick={vi.fn()} disabled={false} />)
    const matchedDiv = container.querySelector(".opacity-70")
    expect(matchedDiv).toBeInTheDocument()
  })

  it("applies rotate class when revealed", () => {
    const flippedCard: CardData = { ...baseCard, flipped: true }
    const { container } = render(<Card card={flippedCard} index={0} onClick={vi.fn()} disabled={false} />)
    const rotatedDiv = container.querySelector(".rotate-y-180")
    expect(rotatedDiv).toBeInTheDocument()
  })
})
