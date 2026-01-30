document.addEventListener("DOMContentLoaded", () => {
  const bookingForm = document.getElementById("booking-form");

  bookingForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    console.log("Submitting booking form...");

    const bookingDate = document.getElementById("booking-date").value;
    const preferredTime = document.getElementById("preferred-time").value;
    const serviceType = document.getElementById("service-type").value;
    const bookingMessage = document.getElementById("booking-message").value;

    const laborerName =
      document
        .querySelector(".laborer-card .btn-primary.clicked")
        ?.closest(".laborer-card")
        ?.querySelector("h3")?.innerText || "Unknown";

    const userEmail = localStorage.getItem("userEmail") || "guest@example.com";

    const formData = new FormData();
    formData.append("laborer_name", laborerName);
    formData.append("user_email", userEmail);
    formData.append("booking_date", bookingDate);
    formData.append("time_slot", preferredTime);
    formData.append("service_type", serviceType);
    formData.append("message", bookingMessage);

    try {
      const res = await fetch("http://127.0.0.1:8000/labor/book/", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        alert("✅ Booking successful!");
        bookingForm.reset();
        document.getElementById("booking-modal").classList.add("hidden");
      } else {
        alert("❌ Failed to book: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      alert("Error connecting to server.");
      console.error(err);
    }
  });
});

// Open booking modal for labor cards
document.querySelectorAll(".laborer-card .btn-primary").forEach((button) => {
  button.addEventListener("click", (e) => {
    // Remove 'clicked' from any previously clicked button
    document
      .querySelectorAll(".laborer-card .btn-primary")
      .forEach((btn) => btn.classList.remove("clicked"));

    // Mark this button as clicked
    button.classList.add("clicked");

    // Open the booking modal
    document.getElementById("booking-modal").classList.remove("hidden");
  });
});

// Close the Booking Modal
document.getElementById("close-booking-modal").addEventListener("click", () => {
  document.getElementById("booking-modal").classList.add("hidden");
});
// Smooth scroll to laborers section for hero button
document.getElementById("scroll-to-laborers").addEventListener("click", () => {
  // Scroll to the laborers section smoothly
  document
    .getElementById("laborers-section")
    .scrollIntoView({ behavior: "smooth" });
});

// Toggle FAQ answers
document.querySelectorAll(".faq-question").forEach((faq) => {
  faq.addEventListener("click", () => {
    const answer = faq.nextElementSibling; 
    answer.classList.toggle("hidden"); 
    faq.querySelector(".faq-toggle").textContent = answer.classList.contains(
      "hidden"
    )
      ? "+"
      : "-"; // Update the toggle symbol
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const reviewForm = document.getElementById("review-form");
  const reviewList = document.getElementById("review-list");
  const starRating = document.querySelectorAll(".star-rating span");
  let selectedRating = 0;

  // Enhanced star rating system with animations
  starRating.forEach((star, index) => {
    star.addEventListener("click", () => {
      selectedRating = star.dataset.value;
      document.getElementById("review-rating").value = selectedRating;

      // Update visual feedback
      starRating.forEach((s, i) => {
        s.classList.toggle("active", i < selectedRating);
        s.style.animationDelay = `${i * 0.1}s`;
      });
    });

    // Hover effects
    star.addEventListener("mouseenter", () => {
      const hoverValue = star.dataset.value;
      starRating.forEach((s, i) => {
        s.style.color = i < hoverValue ? "#fbbf24" : "#e2e8f0";
        s.style.transform = i < hoverValue ? "scale(1.1)" : "scale(1)";
      });
    });
  });

  // Reset stars on mouse leave
  document.querySelector(".star-rating").addEventListener("mouseleave", () => {
    starRating.forEach((s, i) => {
      s.style.color = i < selectedRating ? "#f59e0b" : "#e2e8f0";
      s.style.transform = i < selectedRating ? "scale(1.05)" : "scale(1)";
    });
  });

  // Enhanced review submission
  reviewForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const name = document.getElementById("review-name").value.trim();
    const rating = document.getElementById("review-rating").value;
    const comment = document.getElementById("review-comment").value.trim();

    if (!name || !rating || !comment) {
      showAlert("Please fill in all fields and select a rating.", "error");
      return;
    }

    // Show loading state
    const submitBtn = reviewForm.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = "Submitting...";
    submitBtn.disabled = true;

    const formData = new FormData();
    formData.append("user_name", name);
    formData.append("rating", rating);
    formData.append("comment", comment);

    try {
      const res = await fetch("http://127.0.0.1:8000/labor/submit_review/", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        showAlert("Review submitted successfully!", "success");

        // Add review to the page with enhanced styling
        addReviewToPage({
          user_name: name,
          rating: parseInt(rating),
          comment: comment,
          created_at: new Date().toLocaleDateString(),
          is_new: true,
        });

        // Reset form
        reviewForm.reset();
        selectedRating = 0;
        starRating.forEach((s) => {
          s.classList.remove("active");
          s.style.color = "#e2e8f0";
          s.style.transform = "scale(1)";
        });

        // Smooth scroll to new review
        setTimeout(() => {
          const newReview = reviewList.querySelector(
            '.review[data-new="true"]'
          );
          if (newReview) {
            newReview.scrollIntoView({ behavior: "smooth", block: "center" });
            newReview.removeAttribute("data-new");
          }
        }, 300);
      } else {
        showAlert("Failed to submit review. Please try again.", "error");
      }
    } catch (err) {
      showAlert("Could not connect to server. Please try again.", "error");
      console.error(err);
    } finally {
      submitBtn.textContent = originalText;
      submitBtn.disabled = false;
    }
  });

  // Enhanced function to add review to page
  function addReviewToPage(review) {
    const reviewEl = document.createElement("div");
    reviewEl.className = "review";
    if (review.is_new) {
      reviewEl.setAttribute("data-new", "true");
      reviewEl.style.animationDelay = "0.2s";
      reviewEl.style.animation = "slideInUp 0.6s ease-out forwards";
    }

    // Generate user avatar initials
    const initials = review.user_name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase();

    // Generate star display
    const starsHtml = generateStars(review.rating);

    
    const isVerified = Math.random() > 0.7; 

    reviewEl.innerHTML = `
      <div class="review-header">
        <div class="review-user">
          <div class="review-avatar">${initials}</div>
          <div class="review-user-info">
            <h4>${review.user_name}${
      isVerified
        ? '<span class="verified-badge"><i class=""></i> </span>'
        : ""
    }</h4>
            <div class="review-date">${formatDate(review.created_at)}</div>
          </div>
        </div>
        <div class="stars">
          ${starsHtml}
          <span class="rating-number">${review.rating}/5</span>
        </div>
      </div>
      <div class="review-content">
        <p>${review.comment}</p>
      </div>

    `;

    // Insert at the beginning of review list (after heading)
    const heading = reviewList.querySelector("h3");
    if (heading) {
      heading.insertAdjacentElement("afterend", reviewEl);
    } else {
      reviewList.appendChild(reviewEl);
    }
  }

  // Generate star display HTML
  function generateStars(rating) {
    let starsHtml = "";
    for (let i = 1; i <= 5; i++) {
      if (i <= rating) {
        starsHtml += '<span class="star-filled">★</span>';
      } else {
        starsHtml += '<span class="star-empty">★</span>';
      }
    }
    return starsHtml;
  }

  // Format date for display
  function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return "Today";
    } else if (diffDays === 1) {
      return "Yesterday";
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else if (diffDays < 30) {
      return `${Math.floor(diffDays / 7)} weeks ago`;
    } else {
      return date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    }
  }

  // Enhanced alert system
  function showAlert(message, type = "info") {
    const alertEl = document.createElement("div");
    alertEl.className = `alert alert-${type}`;
    alertEl.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 1rem 1.5rem;
      border-radius: 12px;
      color: white;
      font-weight: 600;
      z-index: 10000;
      transform: translateX(400px);
      transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
      box-shadow: 0 10px 30px rgba(0,0,0,0.2);
      backdrop-filter: blur(10px);
      max-width: 350px;
    `;

    // Set colors based on type
    const colors = {
      success: "linear-gradient(135deg, #10b981, #34d399)",
      error: "linear-gradient(135deg, #ef4444, #f87171)",
      info: "linear-gradient(135deg, #3b82f6, #60a5fa)",
    };

    alertEl.style.background = colors[type] || colors.info;
    alertEl.innerHTML = `
      <div style="display: flex; align-items: center; gap: 0.8rem;">
        <i class="fas fa-${
          type === "success"
            ? "check-circle"
            : type === "error"
            ? "exclamation-circle"
            : "info-circle"
        }"></i>
        <span>${message}</span>
      </div>
    `;

    document.body.appendChild(alertEl);

    // Animate in
    setTimeout(() => {
      alertEl.style.transform = "translateX(0)";
    }, 100);

    // Auto remove
    setTimeout(() => {
      alertEl.style.transform = "translateX(400px)";
      setTimeout(() => {
        if (alertEl.parentNode) {
          alertEl.parentNode.removeChild(alertEl);
        }
      }, 400);
    }, 4000);
  }

  // Load existing reviews with enhanced display
  async function loadReviews() {
    try {
      // Show loading skeletons
      showLoadingSkeletons();

      const res = await fetch("http://127.0.0.1:8000/labor/reviews/");
      const data = await res.json();

      // Clear loading skeletons
      clearLoadingSkeletons();

      if (data.reviews && data.reviews.length > 0) {
        data.reviews.forEach((review, index) => {
          setTimeout(() => {
            addReviewToPage(review);
          }, index * 100); // Stagger the animations
        });
      } else {
        showEmptyState();
      }
    } catch (err) {
      console.error("Failed to load reviews:", err);
      clearLoadingSkeletons();
      showEmptyState();
    }
  }

  // Show loading skeletons
  function showLoadingSkeletons() {
    const heading = reviewList.querySelector("h3");
    for (let i = 0; i < 3; i++) {
      const skeleton = document.createElement("div");
      skeleton.className = "review-skeleton skeleton-item";
      skeleton.innerHTML = `
        <div class="skeleton-header">
          <div class="skeleton skeleton-avatar"></div>
          <div>
            <div class="skeleton skeleton-name"></div>
            <div class="skeleton skeleton-date" style="height: 12px; width: 80px; margin-top: 4px;"></div>
          </div>
          <div class="skeleton skeleton-stars"></div>
        </div>
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-text"></div>
      `;
      if (heading) {
        heading.insertAdjacentElement("afterend", skeleton);
      }
    }
  }

  // Clear loading skeletons
  function clearLoadingSkeletons() {
    document.querySelectorAll(".skeleton-item").forEach((el) => el.remove());
  }

  // Show empty state
  function showEmptyState() {
    const emptyState = document.createElement("div");
    emptyState.className = "reviews-empty";
    emptyState.innerHTML = `
      <i class="far fa-comments"></i>
      <h4>No reviews yet</h4>
      <p>Be the first to share your experience with our gardening services!</p>
    `;
    reviewList.appendChild(emptyState);
  }

  // Initialize reviews loading
  loadReviews();
});

// Review interaction functions
function likeReview(button) {
  const countSpan = button.querySelector(".count");
  const currentCount = parseInt(countSpan.textContent.match(/\d+/)[0]);
  const icon = button.querySelector("i");

  if (icon.classList.contains("far")) {
    icon.classList.remove("far");
    icon.classList.add("fas");
    countSpan.textContent = `(${currentCount + 1})`;
    button.style.color = "#38a169";
  } else {
    icon.classList.remove("fas");
    icon.classList.add("far");
    countSpan.textContent = `(${currentCount - 1})`;
    button.style.color = "#718096";
  }
}

function replyToReview(button) {
  // Implement reply functionality
  console.log("Reply to review");
}

function shareReview(button) {
  // Implement share functionality
  if (navigator.share) {
    navigator.share({
      title: "Garden Ease Review",
      text: "Check out this review for Garden Ease services!",
      url: window.location.href,
    });
  } else {
    // Fallback to copying URL
    navigator.clipboard.writeText(window.location.href);
    // You could show a toast notification here
  }
}

// Add CSS animations
const style = document.createElement("style");
style.textContent = `
  @keyframes slideInUp {
    from {
      opacity: 0;
      transform: translateY(30px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  .review[data-new="true"] {
    border: 2px solid #38a169;
    box-shadow: 0 0 20px rgba(56, 161, 105, 0.2);
  }
`;
document.head.appendChild(style);
