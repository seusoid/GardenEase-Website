    // Smooth scrolling for explore button
    document.getElementById("explore-btn").addEventListener("click", function () {
      const servicesSection = document.getElementById("services");
      servicesSection.scrollIntoView({ behavior: "smooth" });
    });

    // Navbar scroll effect
    window.addEventListener('scroll', function() {
      const navbar = document.getElementById('navbar');
      if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
      } else {
        navbar.classList.remove('scrolled');
      }
    });

    // Mobile menu toggle
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const navLinks = document.getElementById('navLinks');

    mobileMenuToggle.addEventListener('click', function() {
      navLinks.classList.toggle('active');
      const icon = mobileMenuToggle.querySelector('i');
      if (navLinks.classList.contains('active')) {
        icon.className = 'fas fa-times';
      } else {
        icon.className = 'fas fa-bars';
      }
    });

    // Close mobile menu when clicking on a link
    navLinks.addEventListener('click', function() {
      navLinks.classList.remove('active');
      const icon = mobileMenuToggle.querySelector('i');
      icon.className = 'fas fa-bars';
    });

    // Fade in animation on scroll
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, observerOptions);

    // Observe all fade-in elements
    document.querySelectorAll('.fade-in').forEach(el => {
      observer.observe(el);
    });

    // Keep original localStorage functionality
    const userName = localStorage.getItem("userName");
    const userEmail = localStorage.getItem("userEmail");

    // Show welcome message if user data exists
    if (userName && document.getElementById("welcome")) {
      document.getElementById("welcome").innerText = `Welcome, ${userName}!`;
    }