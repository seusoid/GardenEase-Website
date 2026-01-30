let cartData = [];

async function fetchCartData() {
  const userEmail = localStorage.getItem("userEmail");
  if (!userEmail) {
    alert("Please log in to view your cart.");
    return;
  }

  const response = await fetch(`http://127.0.0.1:8000/products/cart/${userEmail}/`);
  cartData = await response.json();  

  loadCart(cartData);
}

function loadCart(cartData) {
  const cartContainer = document.getElementById("cart-items");
  const totalElem = document.getElementById("cart-total");

  cartContainer.innerHTML = "";
  let total = 0;

  cartData.forEach((item, index) => {
  const row = document.createElement("tr");

  const itemTotal = item.price * item.quantity;
  total += itemTotal;

  row.innerHTML = `
    <td>
      <img src="${item.image}" alt="${item.name}" style="width: 50px; height: 50px; object-fit: cover;">
      <span>${item.name}</span>
    </td>
    <td>
      <input type="number" value="${item.quantity}" class="cart-quantity" data-index="${index}" min="1">
    </td>
    <td>PKR ${item.price}</td>
    <td>PKR ${itemTotal.toFixed(2)}</td>
    <td><button class="remove-btn" data-index="${index}">Remove</button></td>
  `;

  cartContainer.appendChild(row);
  const qtyInput = row.querySelector(".cart-quantity");
qtyInput.addEventListener("change", (e) => {
  const newQty = parseInt(e.target.value);
  if (newQty >= 1) {
    updateQuantity(item.id, newQty);
  }
});

});


  totalElem.textContent = `PKR ${total.toFixed(2)}`;
}


  
  // Handle quantity change
function updateQuantity(itemId, newQuantity) {
  fetch(`http://127.0.0.1:8000/products/cart/update/${itemId}/`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ quantity: newQuantity })
  })
  .then(res => res.json())
  .then(data => {
    console.log("✅ Quantity updated", data);
    fetchCartData(); 
  })
  .catch(err => console.error("❌ Failed to update quantity", err));
}






  
  // Handle item removal
  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("remove-btn")) {
      const index = e.target.getAttribute("data-index");
      const itemId = cartData[index].id;

fetch(`http://127.0.0.1:8000/products/cart/delete/${itemId}/`, {
  method: "DELETE"
})
  .then(res => res.json())
  .then(data => {
    cartData.splice(index, 1);
    loadCart(cartData);
  });

    }
  });
  
  
  
  // Checkout button
  document.getElementById("checkout-btn").addEventListener("click", () => {
    alert("Proceeding to Checkout!");
  });
  

fetchCartData();
  