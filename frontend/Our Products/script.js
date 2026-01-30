
let selectedProductId = null;


document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", async () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const category = tab.getAttribute("data-category");
      const plantContainer = document.getElementById("plant-container");
      plantContainer.innerHTML = "<p>Loading...</p>";

      try {
        const response = await fetch(`http://127.0.0.1:8000/products/${category}/`);
        const plants = await response.json();

        if (!plants.length) {
          plantContainer.innerHTML = "<p>No products found.</p>";
          return;
        }

        plantContainer.innerHTML = plants.map(plant => `
  <div class="plant-card">
    <img src="${plant.image}" alt="${plant.name}">
    <h3>${plant.name}</h3>
    <p>PKR ${plant.price}</p>
    <button 
      class="btn-primary"
      data-id="${plant._id}"
      data-name="${plant.name}"
      data-price="${plant.price}"
      data-image="${plant.image}"
      onclick="openModal(this)"
    >
      Shop Now
    </button>
  </div>
`).join("");


      } catch (err) {
        console.error("Failed to load plants:", err);
        plantContainer.innerHTML = "<p>Error loading products.</p>";
      }
    });
  });





  document.getElementById("close-modal").addEventListener("click", () => {
    document.getElementById("quantity-modal").classList.add("hidden");
  });

  // Load default category
  document.querySelector(".tab.active").click();
});



function openModal(button) {
  selectedProductId = button.dataset.id;
  const productName = button.dataset.name;
  const productPrice = parseFloat(button.dataset.price);

  const modal = document.getElementById("quantity-modal");

  modal.querySelector(".modal-product-name").textContent = productName;
  modal.querySelector(".modal-product-price").textContent = `PKR ${productPrice.toFixed(2)}`;
  modal.querySelector("#quantity-input").value = 1;
  modal.querySelector(".modal-total-price").textContent = `PKR ${productPrice.toFixed(2)}`;
  modal.classList.remove("hidden");

  const quantityInput = modal.querySelector("#quantity-input");
  quantityInput.addEventListener("input", () => {
    const quantity = parseInt(quantityInput.value) || 1;
    const totalPrice = productPrice * quantity;
    modal.querySelector(".modal-total-price").textContent = `PKR ${totalPrice.toFixed(2)}`;
  });

  // ✅ Attach "Add to Cart" button inside modal
  modal.querySelector(".btn-primary").onclick = async () => {
    const userEmail = localStorage.getItem("userEmail");
    const quantity = parseInt(modal.querySelector("#quantity-input").value);

    if (!userEmail || !selectedProductId || !quantity) {
      alert("Missing user or product data.");
      return;
    }

    const response = await fetch("http://127.0.0.1:8000/products/add/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_email: userEmail,
        product_id: selectedProductId,
        quantity: quantity
      })
    });

    const data = await response.json();
    if (response.ok) {
      alert(data.message || "Added to cart!");
      modal.classList.add("hidden");
    } else {
      alert("Failed to add to cart.");
    }
  };
}



