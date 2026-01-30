document.getElementById("upload-image-btn").addEventListener("click", () => {
    document.getElementById("image-input").click();
});

document.getElementById("image-input").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) {
        const img = document.getElementById("uploaded-image");
        img.src = URL.createObjectURL(file);
        document.getElementById("output-container").classList.remove("hidden");

        // Prepare form data
        const formData = new FormData();
        formData.append('image', file);

        // Send to backend Django endpoint
        try {
            const response = await fetch('/plant/detect/', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.class_name) {
                document.querySelector(".plant-details h3").innerText = "Detected Plant";
                document.querySelector(".plant-details p:nth-of-type(1)").innerText = data.class_name;
                document.querySelector(".plant-details p:nth-of-type(2)").innerText = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            } else {
                document.querySelector(".plant-details h3").innerText = "Detection Failed";
                document.querySelector(".plant-details p:nth-of-type(1)").innerText = "";
                document.querySelector(".plant-details p:nth-of-type(2)").innerText = "";
            }
        } catch (error) {
            console.error("Error:", error);
            document.querySelector(".plant-details h3").innerText = "Detection Error";
            document.querySelector(".plant-details p:nth-of-type(1)").innerText = "";
            document.querySelector(".plant-details p:nth-of-type(2)").innerText = "";
        }
    }
});
