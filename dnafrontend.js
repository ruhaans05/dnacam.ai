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
  video.style.display = 'block';
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  setTimeout(() => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    video.srcObject.getTracks().forEach(track => track.stop());
    video.style.display = 'none';

    canvas.toBlob(blob => {
      currentImageBlob = blob;
      statusText.textContent = "📷 Captured from webcam.";
    }, 'image/jpeg');
  }, 3000); // Capture after 3 sec
});

// Image upload
uploadInput.addEventListener('change', () => {
  const file = uploadInput.files[0];
  if (file) {
    currentImageBlob = file;
    statusText.textContent = "✅ Image uploaded.";
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
    const response = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (data.result) {
      resultBox.textContent = data.result;
      statusText.textContent = "✅ Done!";
    } else {
      resultBox.textContent = "❌ Error: " + data.error;
      statusText.textContent = "⚠️ Failed.";
    }
  } catch (err) {
    resultBox.textContent = "❌ Error connecting to server.";
    statusText.textContent = "⚠️ Failed.";
  }
});
