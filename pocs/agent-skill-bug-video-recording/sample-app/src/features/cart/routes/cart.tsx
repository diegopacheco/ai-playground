import { QtyStepper } from '../components/QtyStepper'

export function CartPage() {
  return (
    <section className="page">
      <h1>Cart</h1>
      <div className="cart-list">
        <QtyStepper label="Ceramic Pour-Over Coffee Dripper Cone Set" />
        <QtyStepper label="Stainless Steel Insulated Water Bottle 750ml" />
      </div>
    </section>
  )
}
