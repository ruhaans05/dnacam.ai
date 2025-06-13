console.log("✅ JavaScript loaded");

const uploadInput = document.getElementById('upload');
const captureBtn = document.getElementById('captureBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const statusText = document.getElementById('status');
const resultBox = document.getElementById('resultBox');

let currentImageBlob = null;
let stream = null;

// Webcam capture
captureBtn.addEventListener('click', async () => {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Webcam not supported in this browser.");
      return;
    }

    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = 'block';
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
          statusText.textContent = "❌ Could not capture image.";
        } else {
          currentImageBlob = blob;
          statusText.textContent = "✅ Captured from webcam.";
        }
      }, 'image/jpeg');
    }, 3000);
  } catch (err) {
    console.error("Webcam error:", err);
    alert("❌ Webcam access failed. Please allow permission.");
    statusText.textContent = "❌ Webcam error.";
  }
});

// Image upload
uploadInput.addEventListener('change', () => {
  const file = uploadInput.files[0];
  if (file) {
    currentImageBlob = file;
    statusText.textContent = "✅ Image uploaded.";
  } else {
    statusText.textContent = "❌ No image selected.";
  }
});

// Analyze with backend (use relative path!)
analyzeBtn.addEventListener('click', async () => {
  if (!currentImageBlob) {
    statusText.textContent = "⚠️ Upload or capture an image first.";
    return;
  }

  statusText.textContent = "⏳ Sending to OpenAI...";
  const formData = new FormData();
  formData.append("image", currentImageBlob);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const data = await response.json();

    if (data.result) {
      resultBox.textContent = data.result;
      statusText.textContent = "✅ Done!";
    } else {
      resultBox.textContent = "❌ OpenAI error: " + data.error;
      statusText.textContent = "⚠️ OpenAI failed.";
    }
  } catch (err) {
    console.error("Analyze error:", err);
    resultBox.textContent = "❌ Something went wrong.";
    statusText.textContent = `❌ Failed: ${err.message}`;
  }
});
