import { memo } from "react";

const formatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});

function PriceTag({ amount = 0, currency = "USD" }) {
  const formatted =
    currency === "USD" ? formatter.format(amount) : amount.toFixed(2) + " " + currency;
  return <span className="price-tag">{formatted}</span>;
}

export default memo(PriceTag);
