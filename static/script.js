function showUpload() {
  document.getElementById('uploadForm').style.display = 'block';
  document.getElementById('cameraContainer').style.display = 'none';
}

function startCamera() {
  document.getElementById('uploadForm').style.display = 'none';
  document.getElementById('cameraContainer').style.display = 'block';
  const video = document.getElementById('camera');
  navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
  });
}

function capture() {
  const video = document.getElementById('camera');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  const frame = canvas.toDataURL('image/jpeg', 0.6);
  const formData = new FormData();
  formData.append('frame', frame);

  fetch('/detect', { method: 'POST', body: formData })
    .then(res => res.text())
    .then(html => document.body.innerHTML = html);
}
