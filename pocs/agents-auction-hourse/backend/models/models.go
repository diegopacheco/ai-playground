package models

import "time"

type Auction struct {
	ID        string         `json:"id"`
	Winner    string         `json:"winner"`
	CreatedAt time.Time      `json:"created_at"`
	EndedAt   *time.Time     `json:"ended_at"`
	Agents    []AuctionAgent `json:"agents"`
	Rounds    []Round        `json:"rounds"`
}

type AuctionAgent struct {
	ID              string `json:"id"`
	AuctionID       string `json:"auction_id"`
	AgentName       string `json:"agent_name"`
	Model           string `json:"model"`
	InitialBudget   int    `json:"initial_budget"`
	RemainingBudget int    `json:"remaining_budget"`
	ItemsWon        int    `json:"items_won"`
}

type Round struct {
	ID          string `json:"id"`
	AuctionID   string `json:"auction_id"`
	RoundNumber int    `json:"round_number"`
	ItemName    string `json:"item_name"`
	ItemEmoji   string `json:"item_emoji"`
	WinnerAgent string `json:"winner_agent"`
	WinningBid  int    `json:"winning_bid"`
	Bids        []Bid  `json:"bids"`
}

type Bid struct {
	ID             string `json:"id"`
	RoundID        string `json:"round_id"`
	AuctionID      string `json:"auction_id"`
	AgentName      string `json:"agent_name"`
	Amount         int    `json:"amount"`
	Reasoning      string `json:"reasoning"`
	Fallback       bool   `json:"fallback"`
	RawOutput      string `json:"raw_output"`
	ResponseTimeMs int64  `json:"response_time_ms"`
}

type CreateAuctionRequest struct {
	Agents []AgentConfig `json:"agents"`
}

type AgentConfig struct {
	Name   string `json:"name"`
	Model  string `json:"model"`
	Budget int    `json:"budget"`
}

type FunItem struct {
	Name  string
	Emoji string
}

type SSEEvent struct {
	Event string      `json:"event"`
	Data  any `json:"data"`
}

type RoundStartData struct {
	Round     int    `json:"round"`
	ItemName  string `json:"item"`
	ItemEmoji string `json:"item_emoji"`
}

type AgentThinkingData struct {
	Agent string `json:"agent"`
	Round int    `json:"round"`
}

type AgentBidData struct {
	Agent     string `json:"agent"`
	Bid       int    `json:"bid"`
	Reasoning string `json:"reasoning"`
	Round     int    `json:"round"`
	Fallback  bool   `json:"fallback"`
}

type RoundResultData struct {
	Winner     string `json:"winner"`
	Item       string `json:"item"`
	WinningBid int    `json:"winning_bid"`
	Round      int    `json:"round"`
}

type FinalStanding struct {
	Agent          string `json:"agent"`
	ItemsWon       int    `json:"items_won"`
	TotalSpent     int    `json:"total_spent"`
	RemainingBudge int    `json:"remaining_budget"`
}

type AuctionOverData struct {
	FinalStandings []FinalStanding `json:"final_standings"`
	Winner         string          `json:"winner"`
}
