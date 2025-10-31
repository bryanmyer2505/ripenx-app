from flask import Flask, render_template, request, send_file, Response
from ultralytics import YOLO
from PIL import Image
import io, os, base64, cv2, time, requests
from datetime import datetime
from collections import Counter
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB upload limit

# ---------------- Model Auto-Download ----------------
MODEL_PATH = "models/best.pt"
MODEL_URL = "https://github.com/bryanmyer2505/ripenx-app/releases/download/v1.0-model/best.pt"

# Auto-download model if missing
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("📦 Mengunduh model dari GitHub release...")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("✅ Model berhasil diunduh dan disimpan ke", MODEL_PATH)

# Load YOLOv11 model
print("🧠 Memuat model YOLOv11...")
model = YOLO(MODEL_PATH)
CONF_THRESHOLD = 0.4
print("✅ Model siap digunakan.")

latest_results = []  # store for PDF export
camera = None  # global webcam reference

# ---------------- Tracking & Recording ----------------
is_recording = False
recorder = None
recorder_filepath = None
record_start_time = None
unique_counts = Counter()
tracked_objects = {}
next_object_id = 0
frame_idx = 0
MAX_LOST_FRAMES = 30
IOU_MATCH_THRESHOLD = 0.45
last_frame_for_pdf = None

# ---------------- Utility Functions ----------------
def iou(boxA, boxB):
    """Hitung IoU antara dua bounding box."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def boxes_from_results(results):
    """Ambil hasil deteksi dari model YOLO dengan aman."""
    boxes_obj = results[0].boxes
    if boxes_obj is None:
        return []
    data = getattr(boxes_obj, "data", None)
    if data is None:
        return []
    arr = data.cpu().numpy()
    return arr.tolist() if arr.size else []

# ---------------- PDF Header/Footer ----------------
def add_header_footer(canvas, doc):
    """Tambahkan logo dan footer RipenX di setiap halaman PDF."""
    logo_path = os.path.join("static", "RipenX logo.jpg")
    width, height = A4
    if os.path.exists(logo_path):
        logo = ImageReader(logo_path)
        canvas.drawImage(logo, 40, height - 60, width=40, height=40, mask='auto')
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(90, height - 45, "RipenX AI Vision System – HPI Agro")
    canvas.setStrokeColorRGB(0.3, 0.7, 0.3)
    canvas.setLineWidth(0.5)
    canvas.line(40, 40, width - 40, 40)
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(40, 28, "© 2025 RipenX AI Vision System – Laporan Rahasia")

# ---------------- Flask Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- Single Image Detection ----------------
@app.route("/detect", methods=["POST"])
def detect():
    global latest_results, camera

    if camera and camera.isOpened():
        camera.release()
        camera = None

    latest_results = []
    if "image" not in request.files and "frame" not in request.form:
        return "<p style='color:red'>⚠️ Tidak ada gambar atau frame yang tersedia.</p>"

    images = []
    if "image" in request.files:
        for file in request.files.getlist("image"):
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            images.append((file.filename, img))
    elif "frame" in request.form:
        frame_data = request.form["frame"].split(",")[1]
        img_data = base64.b64decode(frame_data)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(("Live_Capture.png", img))

    results_html = ""
    for filename, img in images:
        results = model.predict(img)
        detections = boxes_from_results(results)
        annotated = results[0].plot()

        img_buffer = io.BytesIO()
        Image.fromarray(annotated).save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        det_summary = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf < CONF_THRESHOLD:
                continue
            label = model.names[int(cls)]
            det_summary.append({"label": label, "confidence": float(conf)})

        latest_results.append({
            "filename": filename,
            "detections": det_summary,
            "image": img_buffer.getvalue()
        })

        warning_html = ""
        if any(d["confidence"] < 0.5 for d in det_summary):
            warning_html = "<div class='alert-warning'>⚠️ Beberapa deteksi memiliki tingkat kepercayaan rendah.</div>"

        det_rows = "".join([
            f"<tr><td>{d['label']}</td><td>{d['confidence']*100:.2f}%</td></tr>"
            for d in det_summary
        ])

        results_html += f"""
        <div class='card'>
            <p><b>File:</b> {filename}</p>
            <img src='data:image/png;base64,{img_base64}' alt='Hasil Deteksi'>
            {warning_html}
            <h4>Rangkuman Deteksi</h4>
            <table class='prob-table'>
                <tr><th>Kelas</th><th>Kepercayaan</th></tr>
                {det_rows}
            </table>
        </div>"""

    return render_template("results.html", results_html=results_html)

# ---------------- Stream Page ----------------
@app.route('/stream')
def stream():
    return render_template('stream.html')

# ---------------- Frame Generator ----------------
def generate_frames():
    """Streaming deteksi langsung dengan pelacakan objek."""
    global camera, is_recording, recorder, unique_counts, tracked_objects, next_object_id, frame_idx, last_frame_for_pdf
    try:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)

        prev_time = 0
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame_idx += 1
            results = model.predict(frame)
            annotated = results[0].plot()
            detections = boxes_from_results(results)

            curr_dets = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf < CONF_THRESHOLD:
                    continue
                label = model.names[int(cls)]
                curr_dets.append({'bbox': [x1, y1, x2, y2], 'label': label})

            # IoU-based tracking
            matched, used = set(), set()
            for obj_id, obj in list(tracked_objects.items()):
                best_i, best_idx = 0, None
                for idx, det in enumerate(curr_dets):
                    if idx in used:
                        continue
                    i = iou(obj['bbox'], det['bbox'])
                    if i > best_i:
                        best_i, best_idx = i, idx
                if best_i >= IOU_MATCH_THRESHOLD and best_idx is not None:
                    tracked_objects[obj_id]['bbox'] = curr_dets[best_idx]['bbox']
                    tracked_objects[obj_id]['last_seen'] = frame_idx
                    matched.add(obj_id)
                    used.add(best_idx)

            for idx, det in enumerate(curr_dets):
                if idx not in used:
                    obj_id = next_object_id
                    tracked_objects[obj_id] = {
                        'bbox': det['bbox'],
                        'class': det['label'],
                        'first_seen': frame_idx,
                        'last_seen': frame_idx
                    }
                    next_object_id += 1
                    used.add(idx)
                    if is_recording:
                        unique_counts[det['label']] += 1

            to_remove = [oid for oid, o in tracked_objects.items() if frame_idx - o['last_seen'] > MAX_LOST_FRAMES]
            for oid in to_remove:
                del tracked_objects[oid]

            if is_recording and recorder:
                last_frame_for_pdf = annotated.copy()
                recorder.write(annotated)

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time else 0
            prev_time = current_time
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)

            ret, buffer = cv2.imencode('.jpg', annotated)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    finally:
        if camera and camera.isOpened():
            camera.release()
            camera = None
            print("📷 Kamera dilepaskan setelah streaming selesai.")

# ---------------- Stream Routes ----------------
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream')
def stop_stream():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None
        print("📷 Kamera dihentikan secara manual.")
        return "Kamera berhasil dihentikan."
    return "Kamera sudah berhenti."

# ---------------- Start Recording ----------------
@app.route('/start_record')
def start_record():
    global is_recording, recorder, unique_counts, record_start_time, recorder_filepath, tracked_objects, next_object_id, frame_idx, camera

    if not camera or not camera.isOpened():
        return "⚠️ Kamera belum aktif. Harap mulai streaming terlebih dahulu."

    if not os.path.exists("recordings"):
        os.makedirs("recordings")

    w, h = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640, int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    tracked_objects.clear()
    next_object_id, frame_idx = 0, 0
    unique_counts = Counter()
    record_start_time = time.time()

    recorder_filepath = f"recordings/session_{int(record_start_time)}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recorder = cv2.VideoWriter(recorder_filepath, fourcc, 20.0, (w, h))
    is_recording = True
    return "Perekaman dimulai."

# ---------------- Stop Recording + Generate PDF ----------------
@app.route('/stop_record')
def stop_record():
    global is_recording, recorder, unique_counts, record_start_time, last_frame_for_pdf
    is_recording = False
    if recorder:
        recorder.release()
        recorder = None

    duration = time.time() - record_start_time if record_start_time else 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"Laporan_RipenX_{file_time}.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Cover page
    logo_path = os.path.join("static", "RipenX logo.jpg")
    if os.path.exists(logo_path):
        story.append(RLImage(logo_path, width=2.5 * inch, height=2.5 * inch))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Laporan Deteksi Buah Kelapa Sawit HPI-Agro</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Informasi Proyek: Sistem Deteksi RipenX", styles["Normal"]))
    story.append(Paragraph(f"Tanggal Dibuat: {timestamp}", styles["Normal"]))
    story.append(Spacer(1, 380))
    story.append(Paragraph("<i>Laporan ini dihasilkan otomatis oleh sistem RipenX</i>", styles["Normal"]))
    story.append(PageBreak())

    # Results page
    story.append(Paragraph("<b>Ringkasan Deteksi Langsung RipenX</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Durasi Perekaman: {duration:.1f} detik", styles["Normal"]))
    story.append(Spacer(1, 12))

    if last_frame_for_pdf is not None:
        img_buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(last_frame_for_pdf, cv2.COLOR_BGR2RGB)).save(img_buf, format='PNG')
        img_buf.seek(0)
        story.append(RLImage(img_buf, width=400, height=300))
        story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Ringkasan Jumlah Objek Terdeteksi</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))

    if unique_counts:
        table_data = [["Kelas", "Jumlah Unik"]] + [[cls, str(cnt)] for cls, cnt in unique_counts.items()]
        table = Table(table_data, colWidths=[200, 100])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4CAF50")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("Tidak ada objek unik yang terdeteksi selama sesi ini.", styles["Normal"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph("Laporan ini dihasilkan secara otomatis oleh sistem RipenX AI Vision.", styles["Italic"]))

    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    return send_file(pdf_path, as_attachment=True)

# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
