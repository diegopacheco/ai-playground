package persistence

import (
	"database/sql"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"agents-auction-house/backend/models"
)

var DB *sql.DB

func InitDB() error {
	var err error
	DB, err = sql.Open("sqlite3", "./auction.db?_journal_mode=WAL")
	if err != nil {
		return err
	}

	_, err = DB.Exec(`
		CREATE TABLE IF NOT EXISTS auctions (
			id TEXT PRIMARY KEY,
			winner TEXT,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			ended_at DATETIME
		);
		CREATE TABLE IF NOT EXISTS auction_agents (
			id TEXT PRIMARY KEY,
			auction_id TEXT,
			agent_name TEXT,
			model TEXT,
			initial_budget INTEGER,
			remaining_budget INTEGER,
			items_won INTEGER DEFAULT 0,
			FOREIGN KEY (auction_id) REFERENCES auctions(id)
		);
		CREATE TABLE IF NOT EXISTS rounds (
			id TEXT PRIMARY KEY,
			auction_id TEXT,
			round_number INTEGER,
			item_name TEXT,
			item_emoji TEXT,
			winner_agent TEXT,
			winning_bid INTEGER,
			FOREIGN KEY (auction_id) REFERENCES auctions(id)
		);
		CREATE TABLE IF NOT EXISTS bids (
			id TEXT PRIMARY KEY,
			round_id TEXT,
			auction_id TEXT,
			agent_name TEXT,
			amount INTEGER,
			reasoning TEXT,
			fallback INTEGER DEFAULT 0,
			raw_output TEXT,
			response_time_ms INTEGER,
			FOREIGN KEY (round_id) REFERENCES rounds(id),
			FOREIGN KEY (auction_id) REFERENCES auctions(id)
		);
	`)
	return err
}

func CreateAuction(auction *models.Auction) error {
	_, err := DB.Exec("INSERT INTO auctions (id, created_at) VALUES (?, ?)", auction.ID, auction.CreatedAt)
	return err
}

func CreateAuctionAgent(agent *models.AuctionAgent) error {
	_, err := DB.Exec(
		"INSERT INTO auction_agents (id, auction_id, agent_name, model, initial_budget, remaining_budget, items_won) VALUES (?, ?, ?, ?, ?, ?, ?)",
		agent.ID, agent.AuctionID, agent.AgentName, agent.Model, agent.InitialBudget, agent.RemainingBudget, agent.ItemsWon,
	)
	return err
}

func CreateRound(round *models.Round) error {
	_, err := DB.Exec(
		"INSERT INTO rounds (id, auction_id, round_number, item_name, item_emoji) VALUES (?, ?, ?, ?, ?)",
		round.ID, round.AuctionID, round.RoundNumber, round.ItemName, round.ItemEmoji,
	)
	return err
}

func UpdateRoundWinner(roundID string, winner string, winningBid int) error {
	_, err := DB.Exec("UPDATE rounds SET winner_agent = ?, winning_bid = ? WHERE id = ?", winner, winningBid, roundID)
	return err
}

func CreateBid(bid *models.Bid) error {
	fallbackInt := 0
	if bid.Fallback {
		fallbackInt = 1
	}
	_, err := DB.Exec(
		"INSERT INTO bids (id, round_id, auction_id, agent_name, amount, reasoning, fallback, raw_output, response_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
		bid.ID, bid.RoundID, bid.AuctionID, bid.AgentName, bid.Amount, bid.Reasoning, fallbackInt, bid.RawOutput, bid.ResponseTimeMs,
	)
	return err
}

func UpdateAgentBudget(agentID string, remainingBudget int, itemsWon int) error {
	_, err := DB.Exec("UPDATE auction_agents SET remaining_budget = ?, items_won = ? WHERE id = ?", remainingBudget, itemsWon, agentID)
	return err
}

func FinishAuction(auctionID string, winner string) error {
	now := time.Now()
	_, err := DB.Exec("UPDATE auctions SET winner = ?, ended_at = ? WHERE id = ?", winner, now, auctionID)
	return err
}

func GetAuction(id string) (*models.Auction, error) {
	auction := &models.Auction{}
	var winner sql.NullString
	var endedAt sql.NullTime
	err := DB.QueryRow("SELECT id, winner, created_at, ended_at FROM auctions WHERE id = ?", id).
		Scan(&auction.ID, &winner, &auction.CreatedAt, &endedAt)
	if err != nil {
		return nil, err
	}
	auction.Winner = winner.String
	if endedAt.Valid {
		auction.EndedAt = &endedAt.Time
	}

	agents, err := GetAuctionAgents(id)
	if err != nil {
		return nil, err
	}
	auction.Agents = agents

	rounds, err := GetAuctionRounds(id)
	if err != nil {
		return nil, err
	}
	auction.Rounds = rounds

	return auction, nil
}

func GetAuctionAgents(auctionID string) ([]models.AuctionAgent, error) {
	rows, err := DB.Query("SELECT id, auction_id, agent_name, model, initial_budget, remaining_budget, items_won FROM auction_agents WHERE auction_id = ?", auctionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var agents []models.AuctionAgent
	for rows.Next() {
		var a models.AuctionAgent
		err := rows.Scan(&a.ID, &a.AuctionID, &a.AgentName, &a.Model, &a.InitialBudget, &a.RemainingBudget, &a.ItemsWon)
		if err != nil {
			return nil, err
		}
		agents = append(agents, a)
	}
	return agents, nil
}

func GetAuctionRounds(auctionID string) ([]models.Round, error) {
	rows, err := DB.Query("SELECT id, auction_id, round_number, item_name, item_emoji, winner_agent, winning_bid FROM rounds WHERE auction_id = ? ORDER BY round_number", auctionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var rounds []models.Round
	for rows.Next() {
		var r models.Round
		var winner sql.NullString
		var winningBid sql.NullInt64
		err := rows.Scan(&r.ID, &r.AuctionID, &r.RoundNumber, &r.ItemName, &r.ItemEmoji, &winner, &winningBid)
		if err != nil {
			return nil, err
		}
		r.WinnerAgent = winner.String
		r.WinningBid = int(winningBid.Int64)

		bids, err := GetRoundBids(r.ID)
		if err != nil {
			return nil, err
		}
		r.Bids = bids
		rounds = append(rounds, r)
	}
	return rounds, nil
}

func GetRoundBids(roundID string) ([]models.Bid, error) {
	rows, err := DB.Query("SELECT id, round_id, auction_id, agent_name, amount, reasoning, fallback, raw_output, response_time_ms FROM bids WHERE round_id = ?", roundID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var bids []models.Bid
	for rows.Next() {
		var b models.Bid
		var fallbackInt int
		err := rows.Scan(&b.ID, &b.RoundID, &b.AuctionID, &b.AgentName, &b.Amount, &b.Reasoning, &fallbackInt, &b.RawOutput, &b.ResponseTimeMs)
		if err != nil {
			return nil, err
		}
		b.Fallback = fallbackInt == 1
		bids = append(bids, b)
	}
	return bids, nil
}

func ListAuctions() ([]models.Auction, error) {
	rows, err := DB.Query("SELECT id, winner, created_at, ended_at FROM auctions ORDER BY created_at DESC")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var auctions []models.Auction
	for rows.Next() {
		var a models.Auction
		var winner sql.NullString
		var endedAt sql.NullTime
		err := rows.Scan(&a.ID, &winner, &a.CreatedAt, &endedAt)
		if err != nil {
			return nil, err
		}
		a.Winner = winner.String
		if endedAt.Valid {
			a.EndedAt = &endedAt.Time
		}

		agents, _ := GetAuctionAgents(a.ID)
		a.Agents = agents

		rounds, _ := GetAuctionRounds(a.ID)
		a.Rounds = rounds

		auctions = append(auctions, a)
	}
	return auctions, nil
}
