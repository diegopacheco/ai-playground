package main

import (
	"math"
	"math/rand"
	"net/http"
	"strconv"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

type CoinFlipResult struct {
	Buckets []int `json:"buckets"`
	Total   int   `json:"total"`
	Flips   int   `json:"flips"`
}

type DiceRollResult struct {
	Averages []float64 `json:"averages"`
	Total    int       `json:"total"`
	Rolls    int       `json:"rolls"`
}

type CLTResult struct {
	SampleMeans []float64 `json:"sampleMeans"`
	SampleSize  int       `json:"sampleSize"`
	NumSamples  int       `json:"numSamples"`
	Source      string    `json:"source"`
}

type GaussianPoint struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

func gaussian(x, mean, stddev float64) float64 {
	return (1.0 / (stddev * math.Sqrt(2*math.Pi))) * math.Exp(-0.5*math.Pow((x-mean)/stddev, 2))
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	r.Use(cors.New(config))

	r.GET("/api/coin-flip", func(c *gin.Context) {
		trials, _ := strconv.Atoi(c.DefaultQuery("trials", "1000"))
		flips, _ := strconv.Atoi(c.DefaultQuery("flips", "100"))

		if trials > 10000 {
			trials = 10000
		}
		if flips > 1000 {
			flips = 1000
		}

		buckets := make([]int, flips+1)
		for i := 0; i < trials; i++ {
			heads := 0
			for j := 0; j < flips; j++ {
				if rand.Float64() < 0.5 {
					heads++
				}
			}
			buckets[heads]++
		}

		c.JSON(http.StatusOK, CoinFlipResult{
			Buckets: buckets,
			Total:   trials,
			Flips:   flips,
		})
	})

	r.GET("/api/dice-roll", func(c *gin.Context) {
		trials, _ := strconv.Atoi(c.DefaultQuery("trials", "1000"))
		rolls, _ := strconv.Atoi(c.DefaultQuery("rolls", "50"))

		if trials > 10000 {
			trials = 10000
		}
		if rolls > 500 {
			rolls = 500
		}

		averages := make([]float64, trials)
		for i := 0; i < trials; i++ {
			sum := 0
			for j := 0; j < rolls; j++ {
				sum += rand.Intn(6) + 1
			}
			averages[i] = float64(sum) / float64(rolls)
		}

		c.JSON(http.StatusOK, DiceRollResult{
			Averages: averages,
			Total:    trials,
			Rolls:    rolls,
		})
	})

	r.GET("/api/clt", func(c *gin.Context) {
		numSamples, _ := strconv.Atoi(c.DefaultQuery("numSamples", "1000"))
		sampleSize, _ := strconv.Atoi(c.DefaultQuery("sampleSize", "30"))
		source := c.DefaultQuery("source", "uniform")

		if numSamples > 10000 {
			numSamples = 10000
		}
		if sampleSize > 500 {
			sampleSize = 500
		}

		sampleMeans := make([]float64, numSamples)
		for i := 0; i < numSamples; i++ {
			sum := 0.0
			for j := 0; j < sampleSize; j++ {
				switch source {
				case "uniform":
					sum += rand.Float64() * 10
				case "exponential":
					sum += rand.ExpFloat64()
				case "bimodal":
					if rand.Float64() < 0.5 {
						sum += rand.NormFloat64()*1 + 3
					} else {
						sum += rand.NormFloat64()*1 + 8
					}
				}
			}
			sampleMeans[i] = sum / float64(sampleSize)
		}

		c.JSON(http.StatusOK, CLTResult{
			SampleMeans: sampleMeans,
			SampleSize:  sampleSize,
			NumSamples:  numSamples,
			Source:      source,
		})
	})

	r.GET("/api/gaussian", func(c *gin.Context) {
		mean, _ := strconv.ParseFloat(c.DefaultQuery("mean", "0"), 64)
		stddev, _ := strconv.ParseFloat(c.DefaultQuery("stddev", "1"), 64)
		points, _ := strconv.Atoi(c.DefaultQuery("points", "200"))

		if stddev <= 0 {
			stddev = 1
		}

		result := make([]GaussianPoint, points)
		start := mean - 4*stddev
		end := mean + 4*stddev
		step := (end - start) / float64(points-1)

		for i := 0; i < points; i++ {
			x := start + float64(i)*step
			result[i] = GaussianPoint{X: x, Y: gaussian(x, mean, stddev)}
		}

		c.JSON(http.StatusOK, result)
	})

	r.Run(":8080")
}
