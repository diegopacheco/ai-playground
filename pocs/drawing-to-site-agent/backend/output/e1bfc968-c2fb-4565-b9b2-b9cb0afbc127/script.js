var products = [
  { id: 1, name: "Wireless Headphones", price: 2.57, icon: "🎧" },
  { id: 2, name: "USB-C Cable", price: 1.29, icon: "🔌" },
  { id: 3, name: "Notebook", price: 4.99, icon: "📓" },
  { id: 4, name: "Desk Lamp", price: 12.49, icon: "💡" },
  { id: 5, name: "Mouse Pad", price: 3.75, icon: "🖱️" },
  { id: 6, name: "Water Bottle", price: 6.00, icon: "🧴" }
];

var cart = [];
var currentQty = 1;
var currentProduct = null;

function showPage(id) {
  document.querySelectorAll(".page").forEach(function (p) {
    p.classList.remove("active");
  });
  document.getElementById(id).classList.add("active");
}

function updateCartBadge() {
  var total = cart.reduce(function (s, i) { return s + i.qty; }, 0);
  var badge = document.getElementById("cart-badge");
  document.getElementById("cart-count").textContent = total;
  badge.classList.toggle("hidden", total === 0);
}

function renderProducts() {
  var list = document.getElementById("product-list");
  list.innerHTML = "";
  products.forEach(function (p) {
    var li = document.createElement("li");
    li.innerHTML =
      '<span class="product-name">' + p.icon + " " + p.name + "</span>" +
      '<span class="product-price">$' + p.price.toFixed(2) + "</span>";
    li.addEventListener("click", function () {
      openDetail(p);
    });
    list.appendChild(li);
  });
}

function openDetail(product) {
  currentProduct = product;
  currentQty = 1;
  document.getElementById("detail-image").textContent = product.icon;
  document.getElementById("detail-name").textContent = product.name;
  document.getElementById("detail-price").textContent = "$" + product.price.toFixed(2);
  document.getElementById("qty-value").textContent = currentQty;
  showPage("page-detail");
}

function renderCheckout() {
  var summary = document.getElementById("cart-summary");
  summary.innerHTML = "";
  var total = 0;
  cart.forEach(function (item) {
    var row = document.createElement("div");
    row.className = "cart-item";
    var lineTotal = item.price * item.qty;
    total += lineTotal;
    row.innerHTML =
      "<span>" + item.name + " × " + item.qty + "</span>" +
      "<span>$" + lineTotal.toFixed(2) + "</span>";
    summary.appendChild(row);
  });
  document.getElementById("checkout-total").textContent = "$" + total.toFixed(2);
}

document.getElementById("login-form").addEventListener("submit", function (e) {
  e.preventDefault();
  renderProducts();
  showPage("page-products");
});

document.getElementById("qty-minus").addEventListener("click", function () {
  if (currentQty > 1) {
    currentQty--;
    document.getElementById("qty-value").textContent = currentQty;
  }
});

document.getElementById("qty-plus").addEventListener("click", function () {
  currentQty++;
  document.getElementById("qty-value").textContent = currentQty;
});

document.getElementById("btn-add-cart").addEventListener("click", function () {
  var existing = cart.find(function (i) { return i.id === currentProduct.id; });
  if (existing) {
    existing.qty += currentQty;
  } else {
    cart.push({
      id: currentProduct.id,
      name: currentProduct.name,
      price: currentProduct.price,
      qty: currentQty
    });
  }
  updateCartBadge();
  renderProducts();
  showPage("page-products");
});

document.getElementById("btn-back-products").addEventListener("click", function () {
  showPage("page-products");
});

document.getElementById("cart-badge").addEventListener("click", function () {
  renderCheckout();
  showPage("page-checkout");
});

document.getElementById("btn-back-shop").addEventListener("click", function () {
  renderProducts();
  showPage("page-products");
});

document.getElementById("checkout-form").addEventListener("submit", function (e) {
  e.preventDefault();
  var orderId = "ORD-" + Date.now().toString(36).toUpperCase();
  document.getElementById("order-id").textContent = "Order #" + orderId;
  cart = [];
  updateCartBadge();
  showPage("page-thankyou");
});

document.getElementById("btn-new-order").addEventListener("click", function () {
  document.getElementById("login-form").reset();
  document.getElementById("checkout-form").reset();
  renderProducts();
  showPage("page-products");
});
