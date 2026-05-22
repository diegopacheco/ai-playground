def free_shipping(total_cents: int, pickup: bool) -> bool:
    if pickup:
        return True
    return total_cents >= 500
