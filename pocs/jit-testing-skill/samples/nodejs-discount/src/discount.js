function freeShipping(totalCents, pickup) {
    if (pickup) {
        return true;
    }
    return totalCents >= 500;
}

module.exports = { freeShipping };
