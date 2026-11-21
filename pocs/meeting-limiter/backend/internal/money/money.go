package money

import "fmt"

type Cents int64

func (c Cents) String() string {
	sign := ""
	v := int64(c)
	if v < 0 {
		sign = "-"
		v = -v
	}
	return fmt.Sprintf("%s$%d.%02d", sign, v/100, v%100)
}

func PerMinutes(hourly Cents, minutes int) Cents {
	return hourly * Cents(minutes) / 60
}
