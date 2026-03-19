package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"agents-auction-house/backend/engine"
	"agents-auction-house/backend/models"
	"agents-auction-house/backend/persistence"
)

func CreateAuction(c *gin.Context) {
	var req models.CreateAuctionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if len(req.Agents) != 3 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "exactly 3 agents required"})
		return
	}

	auctionID := uuid.New().String()
	auction := models.Auction{
		ID:        auctionID,
		CreatedAt: time.Now(),
	}
	if err := persistence.CreateAuction(&auction); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	var auctionAgents []models.AuctionAgent
	for _, ag := range req.Agents {
		budget := ag.Budget
		if budget <= 0 {
			budget = 100
		}
		agent := models.AuctionAgent{
			ID:              uuid.New().String(),
			AuctionID:       auctionID,
			AgentName:       ag.Name,
			Model:           ag.Model,
			InitialBudget:   budget,
			RemainingBudget: budget,
			ItemsWon:        0,
		}
		if err := persistence.CreateAuctionAgent(&agent); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		auctionAgents = append(auctionAgents, agent)
	}

	go engine.RunAuction(auctionID, auctionAgents)

	c.JSON(http.StatusCreated, gin.H{"id": auctionID})
}

func GetAuction(c *gin.Context) {
	id := c.Param("id")
	auction, err := persistence.GetAuction(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "auction not found"})
		return
	}
	c.JSON(http.StatusOK, auction)
}
