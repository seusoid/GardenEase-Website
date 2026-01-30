document.addEventListener("DOMContentLoaded", function () {
  const container = document.getElementById("container");
  const signUpButton = document.getElementById("signUp");
  const signInButton = document.getElementById("signIn");

  // Toggle Panels
  signUpButton.addEventListener("click", () => {
    container.classList.add("right-panel-active");
  });

  signInButton.addEventListener("click", () => {
    container.classList.remove("right-panel-active");
  });

  // Password Visibility Toggle
  document.querySelectorAll(".toggle-password").forEach((button) => {
    button.addEventListener("click", function () {
      const passwordInput = this.previousElementSibling;
      if (passwordInput.type === "password") {
        passwordInput.type = "text";
        this.innerHTML = '<i class="fa fa-eye-slash"></i>';
      } else {
        passwordInput.type = "password";
        this.innerHTML = '<i class="fa fa-eye"></i>';
      }
    });
  });

  // ADD GOOGLE OAUTH HANDLERS
  function handleGoogleAuth() {
    const popup = window.open(
      "http://127.0.0.1:8000/auth/login/google-oauth2/",
      "google-auth",
      "width=500,height=600,scrollbars=yes,resizable=yes"
    );

    // Listen for messages from the popup
    const handleMessage = (event) => {
      if (event.origin !== "http://127.0.0.1:5500") return;

      if (event.data.type === "GOOGLE_AUTH_SUCCESS") {
        popup.close();

        // Store user data
        localStorage.setItem("userEmail", event.data.user.email);
        localStorage.setItem("userName", event.data.user.name);
        localStorage.setItem("email", event.data.user.email);
        localStorage.setItem("is_admin", event.data.user.is_admin);

        alert("Google login successful! Redirecting...");
        window.location.href = "../Index/index.html";

        window.removeEventListener("message", handleMessage);
      } else if (event.data.type === "GOOGLE_AUTH_ERROR") {
        popup.close();
        alert("Google authentication failed. Please try again.");
        window.removeEventListener("message", handleMessage);
      }
    };

    window.addEventListener("message", handleMessage);
  }

  // Google Sign In
  document
    .getElementById("google-signin")
    .addEventListener("click", handleGoogleAuth);

  // Google Sign Up (same as sign in for OAuth)
  document
    .getElementById("google-signup")
    .addEventListener("click", handleGoogleAuth);

  // Signup Form Submission
  document
    .getElementById("signup-form")
    .addEventListener("submit", async function (event) {
      event.preventDefault();

      const name = document.getElementById("signup-name").value.trim();
      const email = document.getElementById("signup-email").value.trim();
      const password = document.getElementById("signup-password").value;

      try {
        const response = await fetch("http://127.0.0.1:8000/userauth/signup/", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, email, password }),
        });

        const data = await response.json();
        if (response.ok) {
          alert("Signup successful! Please log in.");
          document.getElementById("signIn").click();
        } else {
          alert(`Error: ${data.error}`);
        }
      } catch (error) {
        alert("Signup failed. Please try again.");
      }
    });

  // Login Form Submission
  document
    .getElementById("login-form")
    .addEventListener("submit", async function (event) {
      event.preventDefault();

      const email = document.getElementById("login-email").value.trim();
      const password = document.getElementById("login-password").value.trim();

      if (!email || !password) {
        alert("Both email and password are required.");
        return;
      }

      try {
        const response = await fetch("http://127.0.0.1:8000/userauth/login/", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ email, password }),
        });

        const data = await response.json();

        if (response.ok) {
          localStorage.setItem("userEmail", data.email);
          localStorage.setItem("userName", data.name);
          localStorage.setItem("email", data.email);
          localStorage.setItem("is_admin", data.is_admin);

          alert("Login successful! Redirecting...");
          window.location.href = "../Index/index.html";
        } else {
          alert("Error: " + data.error);
        }
      } catch (error) {
        console.error("Login Error:", error);
        alert("Login failed. Please try again.");
      }
    });
});

window.addEventListener('message', (evt) => {
  if (evt.origin !== 'http://127.0.0.1:8000') return; // security check

  if (evt.data.type === 'GOOGLE_AUTH_SUCCESS') {
    const user = evt.data.user;
    // e.g. store user in localStorage
  }
  else if (evt.data.type === 'GOOGLE_AUTH_ERROR') {
    console.error('OAuth error:', evt.data.error);
  }
});


// ADD GOOGLE OAUTH HANDLERS
function handleGoogleAuth() {
  const popup = window.open(
    "http://127.0.0.1:8000/auth/login/google-oauth2/",
    "google-auth",
    "width=500,height=600,scrollbars=yes,resizable=yes"
  );

  // Listen for messages from the popup
  const handleMessage = (event) => {
    // Check origin for security
    if (event.origin !== "http://127.0.0.1:8000") return;

    if (event.data.type === "GOOGLE_AUTH_SUCCESS") {
      popup.close();

      // Store user data
      const user = event.data.user;
      if (user.email) {
        localStorage.setItem("userEmail", user.email);
        localStorage.setItem("userName", user.name);
        localStorage.setItem("email", user.email);
        localStorage.setItem("is_admin", user.is_admin || false);

        alert("Google login successful! Redirecting...");
        
        // Redirect to your landing page (served by Django)
        window.location.href = "http://127.0.0.1:8000/";
      } else {
        alert("Login successful but user data is incomplete. Please try again.");
      }

      window.removeEventListener("message", handleMessage);
    } else if (event.data.type === "GOOGLE_AUTH_ERROR") {
      popup.close();
      alert("Google authentication failed. Please try again.");
      window.removeEventListener("message", handleMessage);
    }
  };

  window.addEventListener("message", handleMessage);

  // Check if popup was blocked
  if (!popup || popup.closed || typeof popup.closed == 'undefined') {
    alert('Popup blocked! Please allow popups for this site.');
  }
}