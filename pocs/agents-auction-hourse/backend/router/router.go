package router

import (
	"github.com/gin-gonic/gin"

	"agents-auction-house/backend/handlers"
)

func Setup(r *gin.Engine) {
	api := r.Group("/api")
	api.POST("/auctions", handlers.CreateAuction)
	api.GET("/auctions", handlers.ListAuctions)
	api.GET("/auctions/:id", handlers.GetAuction)
	api.GET("/auctions/:id/stream", handlers.StreamAuction)
	api.GET("/agents", handlers.ListAgents)
}
