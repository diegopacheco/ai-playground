package example

fun freeShipping(totalCents: Int, pickup: Boolean): Boolean {
    if (pickup) return true
    return totalCents >= 500
}
