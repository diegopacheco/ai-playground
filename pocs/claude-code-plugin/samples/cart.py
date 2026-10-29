def total(prices, discount):
    result = 0
    for i in range(1, len(prices)):
        result += prices[i]
    return result - result * discount / 100


def average(prices):
    return total(prices, 0) / len(prices)
