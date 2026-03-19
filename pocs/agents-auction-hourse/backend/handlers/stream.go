package handlers

import (
	"io"

	"github.com/gin-gonic/gin"

	"agents-auction-house/backend/sse"
)

func StreamAuction(c *gin.Context) {
	auctionID := c.Param("id")

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	ch := sse.Global.Subscribe(auctionID)
	defer sse.Global.Unsubscribe(auctionID, ch)

	c.Stream(func(w io.Writer) bool {
		select {
		case msg, ok := <-ch:
			if !ok {
				return false
			}
			c.Writer.WriteString(msg)
			c.Writer.Flush()
			return true
		case <-c.Request.Context().Done():
			return false
		}
	})
}
