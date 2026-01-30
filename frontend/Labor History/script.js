document.addEventListener("DOMContentLoaded", () => {
  const bookingsGrid = document.getElementById("bookings-grid");
  const totalBookingsElement = document.getElementById("total-bookings");
  const email = localStorage.getItem("email");
  const isAdmin = localStorage.getItem("is_admin") === "true";

    if (!isAdmin) {
    document.body.innerHTML = `
      <div class="unauthorized">
        <h2>Not Authorized</h2>
        <p>You don't have permission to view this page.</p>
      </div>`;
    return;  // stop any further JS
  }

  let bookings = [];

  fetch("/labor/booking-history/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email })
  })
  .then(res => res.json().then(json => ({ ok: res.ok, json })))
  .then(({ ok, json }) => {
    if (!ok) throw new Error(json.error || "Failed to load bookings");
    bookings = json.bookings;       // ← now filled from server
    renderBookings(bookings);       // ← call your existing render function
  })
  .catch(err => showNotification(err.message, "error"));


  // Function to format date
  function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  }

  // Function to format created at timestamp
  function formatCreatedAt(dateString) {
    const date = new Date(dateString);
    return `Created on ${date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    })} at ${date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    })}`;
  }

  // Function to format service type
  function formatServiceType(serviceType) {
    return serviceType
      .split("-")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  // Function to format time slot
  function formatTimeSlot(timeSlot) {
    return timeSlot.charAt(0).toUpperCase() + timeSlot.slice(1);
  }

  // Function to create booking card
  function createBookingCard(booking, index) {
    const card = document.createElement("div");
    card.className = "booking-card";
    card.style.animationDelay = `${index * 0.1}s`;

    const hasMessage = booking.message && booking.message.trim() !== "";

    card.innerHTML = `
          <div class="booking-header">
            <div class="service-title">${formatServiceType(
              booking.service_type
            )}</div>
            
          </div>
          <div class="booking-details">
            <div class="detail-item">
              <i class="fas fa-user"></i>
              <span><strong>Laborer:</strong> ${booking.laborer_name}</span>
            </div>
            <div class="detail-item">
              <i class="fas fa-envelope"></i>
              <span><strong>User:</strong> ${booking.user_email}</span>
            </div>
            <div class="detail-item">
              <i class="fas fa-calendar-alt"></i>
              <span><strong>Date:</strong> ${formatDate(
                booking.booking_date
              )}</span>
            </div>
            <div class="detail-item">
              <i class="fas fa-clock"></i>
              <span><strong>Time:</strong> ${formatTimeSlot(
                booking.time_slot
              )}</span>
            </div>
          </div>
          ${
            hasMessage
              ? `
            <div class="message-section">
              <div class="message-content">
                <strong>Message:</strong> ${booking.message}
              </div>
            </div>
          `
              : ""
          }
          <div class="created-at">
            <i class="fas fa-plus-circle"></i>
            <span>${formatCreatedAt(booking.created_at)}</span>
          </div>
          <div class="booking-actions">
            
            <button class="btn btn-cancel" onclick="cancelBooking('${
              booking.user_email
            }', '${booking.service_type}')">
              <i class="fas fa-times"></i>
              Remove Booking
            </button>
          </div>
        `;

    return card;
  }

  // Function to create empty state
  function createEmptyState() {
    const emptyDiv = document.createElement("div");
    emptyDiv.className = "empty-state";
    emptyDiv.innerHTML = `
          <i class="fas fa-calendar-times"></i>
          <h3>No Bookings Found</h3>
          <p>No bookings match your current filters.</p>
        `;
    return emptyDiv;
  }

  // Function to render bookings
  function renderBookings(bookingsToRender = bookings) {
    bookingsGrid.innerHTML = "";
    totalBookingsElement.textContent = bookingsToRender.length;

    if (bookingsToRender.length === 0) {
      bookingsGrid.appendChild(createEmptyState());
    } else {
      bookingsToRender.forEach((booking, index) => {
        bookingsGrid.appendChild(createBookingCard(booking, index));
      });
    }
  }

  // Function to filter bookings
  function filterBookings() {
    const serviceFilter = document.getElementById("service-filter").value;
    const timeFilter = document.getElementById("time-filter").value;
    const dateFilter = document.getElementById("date-filter").value;

    let filteredBookings = bookings.filter((booking) => {
      const serviceMatch =
        !serviceFilter || booking.service_type === serviceFilter;
      const timeMatch = !timeFilter || booking.time_slot === timeFilter;
      const dateMatch =
        !dateFilter || booking.booking_date.startsWith(dateFilter);

      return serviceMatch && timeMatch && dateMatch;
    });

    renderBookings(filteredBookings);
  }

  // Add event listeners for filters
  document
    .getElementById("service-filter")
    .addEventListener("change", filterBookings);
  document
    .getElementById("time-filter")
    .addEventListener("change", filterBookings);
  document
    .getElementById("date-filter")
    .addEventListener("change", filterBookings);

  // Initial render
  renderBookings();
});

// Global functions for button actions
function viewBookingDetails(userEmail) {
  alert(
    `Viewing details for booking by ${userEmail}.\n\nIn a real application, this would open a detailed modal or navigate to a detailed page with complete booking information.`
  );
}

function cancelBooking(userEmail, serviceType) {
   if (!confirm(`Remove ${formatServiceType(serviceType)} booking for ${userEmail}?`)) return;

  const email = localStorage.getItem("email");

  fetch("/labor/cancel-booking/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email,            // your admin's email
      user_email: userEmail,
      service_type: serviceType  // Send the raw service_type from database
    })
  })
  .then(res => res.json().then(json => ({ ok: res.ok, json })))
  .then(({ ok, json }) => {
    if (!ok) throw new Error(json.error || "Cancel failed");
    showNotification(json.message || "Booking removed");
    // remove from UI:
    const remaining = bookings.filter(b =>
      !(b.user_email === userEmail && b.service_type === serviceType)
    );
    renderBookings(remaining);
    bookings = remaining;  // update our local copy
  })
  .catch(err => showNotification(err.message, "error"));
}

// Helper function to format service type (moved outside to be accessible globally)
function formatServiceType(serviceType) {
  return serviceType
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

// Simple notification system
function showNotification(message, type = "success") {
  const notification = document.createElement("div");
  notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === "success" ? "#c0eb34" : "#dc3545"};
        color: ${type === "success" ? "#112a12" : "white"};
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        z-index: 1000;
        font-weight: 600;
        animation: slideInRight 0.3s ease-out;
      `;
  notification.textContent = message;

  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.animation = "slideOutRight 0.3s ease-out";
    setTimeout(() => {
      notification.remove();
    }, 300);
  }, 3000);
}

// Add CSS for notifications
const style = document.createElement("style");
style.textContent = `
      @keyframes slideInRight {
        from {
          transform: translateX(100%);
          opacity: 0;
        }
        to {
          transform: translateX(0);
          opacity: 1;
        }
      }
      
      @keyframes slideOutRight {
        from {
          transform: translateX(0);
          opacity: 1;
        }
        to {
          transform: translateX(100%);
          opacity: 0;
        }
      }
    `;