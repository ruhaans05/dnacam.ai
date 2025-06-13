const uploadInput = document.getElementById('upload');
const captureBtn = document.getElementById('captureBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const statusText = document.getElementById('status');
const resultBox = document.getElementById('resultBox');

let currentImageBlob = null;

// Webcam capture
captureBtn.addEventListener('click', async () => {
  try {
    video.style.display = 'block';
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    statusText.textContent = "📸 Capturing in 3 seconds...";

    setTimeout(() => {
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);
      stream.getTracks().forEach(track => track.stop());
      video.style.display = 'none';

      canvas.toBlob(blob => {
        if (!blob) {
          statusText.textContent = "❌ Failed to capture image.";
        } else {
          currentImageBlob = blob;
          statusText.textContent = "✅ Image captured from webcam.";
        }
      }, 'image/jpeg');
    }, 3000);
  } catch (err) {
    console.error("Webcam error:", err);
    statusText.textContent = "❌ Webcam access failed. Please allow camera permission.";
  }
});

// Image upload
uploadInput.addEventListener('change', () => {
  const file = uploadInput.files[0];
  if (file) {
    currentImageBlob = file;
    statusText.textContent = "✅ Image uploaded.";
  } else {
    statusText.textContent = "❌ Failed to load image.";
  }
});

// Analyze with backend
analyzeBtn.addEventListener('click', async () => {
  if (!currentImageBlob) {
    statusText.textContent = "⚠️ Please upload or capture an image first.";
    return;
  }

  statusText.textContent = "⏳ Analyzing...";
  const formData = new FormData();
  formData.append("image", currentImageBlob);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();

    if (data.result) {
      resultBox.textContent = data.result;
      statusText.textContent = "✅ Done!";
    } else {
      resultBox.textContent = "❌ Error: " + data.error;
      statusText.textContent = "⚠️ OpenAI failed to respond.";
    }
  } catch (err) {
    console.error("Analyze error:", err);
    resultBox.textContent = "❌ Something went wrong. Check the console.";
    statusText.textContent = `❌ Failed to connect to backend. (${err.message})`;
  }
});
