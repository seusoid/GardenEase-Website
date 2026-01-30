let posts = {
  sharing: [],
  problems: [],
  design: [],
  suggestions: [],
};

async function fetchPosts() {
  const res = await fetch("http://127.0.0.1:8000/community/get_posts/");
  const data = await res.json();

  posts = {
    sharing: [],
    problems: [],
    design: [],
    suggestions: [],
  };

  for (const post of data.posts) {
    let categoryKey = post.category.toLowerCase();
    if (!posts[categoryKey]) continue; // skip unknown categories

    posts[categoryKey].push({
      id: post.id,
      title: "",
      description: post.text,
      img: post.image_url,
      time: new Date(post.created_at).toLocaleString(),
      replies: post.replies.map((r) => ({
        text: r.text,
        img: r.image_url,
      })),
    });
  }

  //  After populating render them:
  const currentCategory = document
    .querySelector(".tab.active")
    .getAttribute("data-category");
  loadPosts(currentCategory);
}

// Load posts into the appropriate section
function loadPosts(category) {
  const container = document.getElementById("posts-container");
  container.innerHTML = posts[category]
    .map(
      (post, index) => `
          <div class="post-card">
          <img src="${post.img}" alt="${post.title}" class="clickable-post-image" data-img-src="${post.img}">

            <h3>${post.title}</h3>
            <p>${post.description}</p>
            <p><small>${post.time}</small></p>
            <button class="btn-primary reply-btn" data-post-index="${index}">Reply</button>
            <button class="btn-secondary toggle-replies-btn" data-post-index="${index}">
              Show Comments (${post.replies.length})
            </button>
            <div class="replies-container hidden" id="replies-${index}">
              ${post.replies
                .map(
                  (reply, idx) => `
                <div class="reply">
                  <p><strong>${idx + 1}.</strong> ${reply.text}</p>
                  ${
                    reply.img
                      ? `<button class="btn-secondary view-image-btn" data-img-src="${reply.img}">View Image</button>`
                      : ""
                  }
                </div>`
                )
                .join("")}
            </div>
          </div>
        `
    )
    .join("");

  const button = document.getElementById("add-post");
  switch (category) {
    case "problems":
      button.innerHTML = '<i class="fa fa-camera"></i> Share Your Problem';
      break;
    case "design":
      button.innerHTML = '<i class="fa fa-camera"></i> Share Your Design';
      break;
    case "suggestions":
      button.innerHTML = '<i class="fa fa-camera"></i> Request Suggestions';
      break;
    default:
      button.innerHTML = '<i class="fa fa-camera"></i> Share Your Idea';
  }

  // Default load

  // Attach event listeners for reply buttons
  document.querySelectorAll(".reply-btn").forEach((btn) => {
    btn.addEventListener("click", openReplyModal);
  });

  // Attach event listeners for toggle replies buttons
  document.querySelectorAll(".toggle-replies-btn").forEach((btn) => {
    btn.addEventListener("click", toggleReplies);
  });

  // Attach event listeners for view image buttons
  document.querySelectorAll(".view-image-btn").forEach((btn) => {
    btn.addEventListener("click", viewImage);
  });
    // Attach event listeners for view image buttons
  document.querySelectorAll(".view-image-btn").forEach((btn) => {
    btn.addEventListener("click", viewImage);
  });

  // ✅ Add this block here!
  document.querySelectorAll(".clickable-post-image").forEach((img) => {
    img.addEventListener("click", viewImage);
  });

}

// Open the reply modal
function openReplyModal(e) {
  const postIndex = e.target.getAttribute("data-post-index");
  const category = document
    .querySelector(".tab.active")
    .getAttribute("data-category");
  const replyModal = document.getElementById("reply-modal");

  replyModal.setAttribute("data-post-index", postIndex);
  replyModal.setAttribute("data-category", category);

  replyModal.classList.remove("hidden");
}

// Submit a reply
const replyForm = document.getElementById("reply-form");
replyForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const postIndex = document
    .getElementById("reply-modal")
    .getAttribute("data-post-index");
  const category = document
    .getElementById("reply-modal")
    .getAttribute("data-category");
  const postId = posts[category][postIndex].id;

  const replyText = document.getElementById("reply-text").value;
  const replyImageInput = document.getElementById("reply-image");

  const formData = new FormData();
  formData.append("post_id", postId);
  formData.append(
    "user_email",
    localStorage.getItem("userEmail") || "guest@example.com"
  );
  formData.append("text", replyText);
  if (replyImageInput.files[0]) {
    formData.append("image", replyImageInput.files[0]);
  }

  const response = await fetch(
    "http://127.0.0.1:8000/community/create_reply/",
    {
      method: "POST",
      body: formData,
    }
  );

  if (response.ok) {
    await fetchPosts();
    document.getElementById("reply-modal").classList.add("hidden");
    replyForm.reset();
  } else {
    alert("Failed to submit reply");
  }
});

// Toggle the visibility of replies
function toggleReplies(e) {
  const postIndex = e.target.getAttribute("data-post-index");
  const repliesContainer = document.getElementById(`replies-${postIndex}`);

  repliesContainer.classList.toggle("hidden");

  if (repliesContainer.classList.contains("hidden")) {
    e.target.textContent = `Show Comments (${
      posts[
        document.querySelector(".tab.active").getAttribute("data-category")
      ][postIndex].replies.length
    })`;
  } else {
    e.target.textContent = "Hide Comments";
  }
}

// View image in a modal
function viewImage(e) {
  const imgSrc = e.target.getAttribute("data-img-src");
  const imageViewerModal = document.getElementById("image-viewer-modal");
  const imageViewer = document.getElementById("image-viewer");

  imageViewer.src = imgSrc;
  imageViewerModal.classList.remove("hidden");
}

// Close image viewer
const closeImageViewerButton = document.getElementById("close-image-viewer");
closeImageViewerButton.addEventListener("click", () => {
  document.getElementById("image-viewer-modal").classList.add("hidden");
});

// Modal functionality
const modal = document.getElementById("post-modal");
const addPostButton = document.getElementById("add-post");
const closeModalButton = document.getElementById("close-modal");
const replyModalCloseButton = document.getElementById("close-reply-modal");

addPostButton.addEventListener("click", () => {
  modal.classList.remove("hidden");
});

closeModalButton.addEventListener("click", () => {
  modal.classList.add("hidden");
});

replyModalCloseButton.addEventListener("click", () => {
  document.getElementById("reply-modal").classList.add("hidden");
});

// Submit new post
const postForm = document.getElementById("post-form");
postForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const category = document

    .querySelector(".tab.active")
    .getAttribute("data-category");
  const imageInput = document.getElementById("post-image");
  const textInput = document.getElementById("post-text");

  const formData = new FormData();
  formData.append(
    "user_email",
    localStorage.getItem("userEmail") || "guest@example.com"
  );
  const categoryMap = {
    sharing: "sharing",
    problems: "problems",
    design: "design",
    suggestions: "suggestions",
  };

  formData.append("category", categoryMap[category] || "Ideas"); // fallback to "Ideas"

  formData.append("text", textInput.value);
  if (imageInput.files[0]) {
    formData.append("image", imageInput.files[0]);
  }

  const response = await fetch("http://127.0.0.1:8000/community/create_post/", {
    method: "POST",
    body: formData,
  });

  if (response.ok) {
    await fetchPosts();
    const category = document
      .querySelector(".tab.active")
      .getAttribute("data-category");
    loadPosts(category);
    alert("Your post has been shared successfully!");
    modal.classList.add("hidden");
    postForm.reset();
  } else {
    alert("Failed to submit post");
  }
});

// Tab switching
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document
      .querySelectorAll(".tab")
      .forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const category = tab.getAttribute("data-category");
    fetchPosts();
  });
});

// Default load
fetchPosts();

const carousel = document.querySelector(".carousel");

// Clone images to create an infinite scroll effect
const images = Array.from(carousel.children);
images.forEach((img) => {
  const clone = img.cloneNode(true);
  carousel.appendChild(clone);
});


