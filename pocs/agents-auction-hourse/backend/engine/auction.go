package engine

import (
	"fmt"
	"log"
	"math/rand"
	"sort"
	"time"

	"github.com/google/uuid"

	"agents-auction-house/backend/agents"
	"agents-auction-house/backend/models"
	"agents-auction-house/backend/persistence"
	"agents-auction-house/backend/sse"
)

var FunItems = []models.FunItem{
	{Name: "Ancient Dragon Egg", Emoji: "\U0001F95A\U0001F409"},
	{Name: "Time-Travel Microwave", Emoji: "\U0001F550\U0001F525"},
	{Name: "Haunted Rubber Duck", Emoji: "\U0001F986\U0001F47B"},
	{Name: "Quantum Burrito", Emoji: "\U0001F32F\u269B\uFE0F"},
	{Name: "Invisible Bicycle", Emoji: "\U0001F6B2\U0001F47B"},
	{Name: "Singing Cactus", Emoji: "\U0001F335\U0001F3A4"},
	{Name: "Telepathic Toaster", Emoji: "\U0001F35E\U0001F9E0"},
	{Name: "Unicorn Roller Skates", Emoji: "\u26F8\uFE0F\U0001F984"},
	{Name: "Gravity-Defying Hammock", Emoji: "\U0001FAA2\U0001F680"},
	{Name: "Exploding Pinata of Wisdom", Emoji: "\U0001FAA5\U0001F4A5"},
	{Name: "Cursed Disco Ball", Emoji: "\U0001FAA9\U0001F608"},
	{Name: "Levitating Pizza Slice", Emoji: "\U0001F355\U0001F6F8"},
	{Name: "Mind-Reading Sunglasses", Emoji: "\U0001F576\uFE0F\U0001F9E0"},
	{Name: "Portal Gun (Slightly Used)", Emoji: "\U0001F52B\U0001F300"},
	{Name: "Shrink Ray Flashlight", Emoji: "\U0001F526\U0001F52C"},
	{Name: "Enchanted USB Cable (Always Right Side Up)", Emoji: "\U0001F50C\u2728"},
	{Name: "Self-Driving Shopping Cart", Emoji: "\U0001F6D2\U0001F916"},
	{Name: "Infinite Coffee Mug", Emoji: "\u2615\u267E\uFE0F"},
	{Name: "Philosophers Stone (Cracked)", Emoji: "\U0001F48E\U0001F528"},
	{Name: "Mass Destruction Cat", Emoji: "\U0001F431\U0001F4A3"},
}

func pickItems(count int) []models.FunItem {
	indices := rand.Perm(len(FunItems))
	var picked []models.FunItem
	for i := 0; i < count && i < len(indices); i++ {
		picked = append(picked, FunItems[indices[i]])
	}
	return picked
}

func RunAuction(auctionID string, auctionAgents []models.AuctionAgent) {
	time.Sleep(1 * time.Second)
	items := pickItems(3)
	totalRounds := 3

	budgets := make(map[string]int)
	agentModels := make(map[string]string)
	agentIDs := make(map[string]string)
	itemsWon := make(map[string]int)
	totalSpent := make(map[string]int)

	for _, a := range auctionAgents {
		budgets[a.AgentName] = a.RemainingBudget
		agentModels[a.AgentName] = a.Model
		agentIDs[a.AgentName] = a.ID
		itemsWon[a.AgentName] = 0
		totalSpent[a.AgentName] = 0
	}

	var roundHistory []models.Round

	for roundIdx := 0; roundIdx < totalRounds; roundIdx++ {
		item := items[roundIdx]
		roundNumber := roundIdx + 1
		roundsLeft := totalRounds - roundNumber

		roundID := uuid.New().String()
		round := models.Round{
			ID:          roundID,
			AuctionID:   auctionID,
			RoundNumber: roundNumber,
			ItemName:    item.Name,
			ItemEmoji:   item.Emoji,
		}
		persistence.CreateRound(&round)

		sse.Global.Send(auctionID, models.SSEEvent{
			Event: "round_start",
			Data: models.RoundStartData{
				Round:     roundNumber,
				ItemName:  item.Name,
				ItemEmoji: item.Emoji,
			},
		})

		var roundBids []models.Bid
		currentHighest := 0

		for agentIdx, agent := range auctionAgents {
			agentName := agent.AgentName

			sse.Global.Send(auctionID, models.SSEEvent{
				Event: "agent_thinking",
				Data: models.AgentThinkingData{
					Agent: agentName,
					Round: roundNumber,
				},
			})

			prompt := buildPrompt(agentName, budgets, roundsLeft, item, roundBids, roundHistory, auctionAgents)

			runner := agents.GetRunner(agentName, agentModels[agentName])
			rawOutput, responseTime, err := runner.Run(prompt)
			if err != nil {
				log.Printf("Agent %s error: %v, output: %s", agentName, err, rawOutput)
			}

			isFirst := agentIdx == 0
			bidAmount, reasoning, fallback := ParseBid(rawOutput, isFirst, currentHighest, budgets[agentName])

			bid := models.Bid{
				ID:             uuid.New().String(),
				RoundID:        roundID,
				AuctionID:      auctionID,
				AgentName:      agentName,
				Amount:         bidAmount,
				Reasoning:      reasoning,
				Fallback:       fallback,
				RawOutput:      rawOutput,
				ResponseTimeMs: responseTime,
			}
			persistence.CreateBid(&bid)
			roundBids = append(roundBids, bid)

			if bidAmount > currentHighest {
				currentHighest = bidAmount
			}

			sse.Global.Send(auctionID, models.SSEEvent{
				Event: "agent_bid",
				Data: models.AgentBidData{
					Agent:     agentName,
					Bid:       bidAmount,
					Reasoning: reasoning,
					Round:     roundNumber,
					Fallback:  fallback,
				},
			})
		}

		winner := determineWinner(roundBids)
		round.WinnerAgent = winner
		round.WinningBid = getBidAmount(roundBids, winner)
		round.Bids = roundBids
		persistence.UpdateRoundWinner(roundID, winner, round.WinningBid)

		budgets[winner] -= round.WinningBid
		itemsWon[winner]++
		totalSpent[winner] += round.WinningBid
		persistence.UpdateAgentBudget(agentIDs[winner], budgets[winner], itemsWon[winner])

		roundHistory = append(roundHistory, round)

		sse.Global.Send(auctionID, models.SSEEvent{
			Event: "round_result",
			Data: models.RoundResultData{
				Winner:     winner,
				Item:       item.Name,
				WinningBid: round.WinningBid,
				Round:      roundNumber,
			},
		})
	}

	auctionWinner := determineAuctionWinner(auctionAgents, itemsWon, totalSpent)
	persistence.FinishAuction(auctionID, auctionWinner)

	var standings []models.FinalStanding
	for _, a := range auctionAgents {
		standings = append(standings, models.FinalStanding{
			Agent:           a.AgentName,
			ItemsWon:        itemsWon[a.AgentName],
			TotalSpent:      totalSpent[a.AgentName],
			RemainingBudget: budgets[a.AgentName],
			InitialBudget:   a.InitialBudget,
		})
	}

	sse.Global.Send(auctionID, models.SSEEvent{
		Event: "auction_over",
		Data: models.AuctionOverData{
			FinalStandings: standings,
			Winner:         auctionWinner,
		},
	})
}

func determineWinner(bids []models.Bid) string {
	winner := bids[0]
	for _, b := range bids[1:] {
		if b.Amount > winner.Amount {
			winner = b
		}
	}
	return winner.AgentName
}

func getBidAmount(bids []models.Bid, agentName string) int {
	for _, b := range bids {
		if b.AgentName == agentName {
			return b.Amount
		}
	}
	return 0
}

func determineAuctionWinner(auctionAgents []models.AuctionAgent, itemsWon map[string]int, totalSpent map[string]int) string {
	type agentScore struct {
		name       string
		items      int
		spent      int
	}
	var scores []agentScore
	for _, a := range auctionAgents {
		scores = append(scores, agentScore{
			name:  a.AgentName,
			items: itemsWon[a.AgentName],
			spent: totalSpent[a.AgentName],
		})
	}
	sort.Slice(scores, func(i, j int) bool {
		if scores[i].items != scores[j].items {
			return scores[i].items > scores[j].items
		}
		return scores[i].spent < scores[j].spent
	})
	return scores[0].name
}

func buildPrompt(agentName string, budgets map[string]int, roundsLeft int, item models.FunItem, currentBids []models.Bid, roundHistory []models.Round, allAgents []models.AuctionAgent) string {
	myBudget := budgets[agentName]

	otherBudgets := ""
	for _, a := range allAgents {
		if a.AgentName != agentName {
			if otherBudgets != "" {
				otherBudgets += ", "
			}
			otherBudgets += fmt.Sprintf("%s=$%d", a.AgentName, budgets[a.AgentName])
		}
	}

	bidsSoFar := ""
	if len(currentBids) > 0 {
		bidsSoFar = "Bids so far this round:\n"
		for _, b := range currentBids {
			bidsSoFar += fmt.Sprintf("- %s bid $%d (\"%s\")\n", b.AgentName, b.Amount, b.Reasoning)
		}
	}

	history := ""
	if len(roundHistory) > 0 {
		history = "Previous round results:\n"
		for _, r := range roundHistory {
			history += fmt.Sprintf("Round %d: %s won %s for $%d\n", r.RoundNumber, r.WinnerAgent, r.ItemName, r.WinningBid)
		}
	}

	return fmt.Sprintf(`You are bidding in a fun auction. You have $%d auction dollars left. There are %d rounds remaining after this one.

ITEM UP FOR BID: %s %s

Your remaining budget: $%d
Other agents budgets: %s

%s
%s
RULES:
- Bid an integer amount you can afford
- Goal: win the most items while spending the least
- Winner = most items won, tiebreaker = least total spent
- You can see what others bid before you (sequential auction)

Respond with ONLY a JSON object, nothing else:
{"bid": <integer>, "reasoning": "<one sentence strategy>"}`, myBudget, roundsLeft, item.Name, item.Emoji, myBudget, otherBudgets, bidsSoFar, history)
}
