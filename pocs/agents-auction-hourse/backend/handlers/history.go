package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"agents-auction-house/backend/agents"
	"agents-auction-house/backend/models"
	"agents-auction-house/backend/persistence"
)

func ListAuctions(c *gin.Context) {
	auctions, err := persistence.ListAuctions()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if auctions == nil {
		auctions = []models.Auction{}
	}
	c.JSON(http.StatusOK, auctions)
}

func ListAgents(c *gin.Context) {
	c.JSON(http.StatusOK, agents.AvailableAgents)
}
