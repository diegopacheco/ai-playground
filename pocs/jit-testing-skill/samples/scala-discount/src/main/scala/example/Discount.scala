package example

object Discount:
  def freeShipping(totalCents: Int, pickup: Boolean): Boolean =
    if pickup then true
    else totalCents >= 500
