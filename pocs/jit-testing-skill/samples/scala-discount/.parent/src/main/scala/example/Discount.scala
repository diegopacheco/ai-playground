package example

object Discount:
  def freeShipping(totalCents: Int, pickup: Boolean): Boolean =
    totalCents >= 5000
