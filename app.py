import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import re
import os
import tempfile
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

st.title("📸 Araç Plaka Tanıma Sistemi")
st.write("YOLOv8 + EasyOCR + DeepSORT ile plaka tanıma ve takip sistemi")

uploaded_file = st.file_uploader("Bir video dosyası yükleyin", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Geçici dosyaya video kayıt
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name

    # Modelleri yükleme
    model = YOLO("best.pt")
    reader = easyocr.Reader(['en'])
    tracker = DeepSort(max_age=1)

    # Plaka listesi
    valid_plates = set()
    if os.path.exists("plakalar.txt"):
        with open("plakalar.txt", "r") as f:
            valid_plates = {
                line.strip().upper().replace(" ", "")
                for line in f
                if line.strip()  # boş satırları atla
            }
        st.success(f"✔️ {len(valid_plates)} plaka yüklendi.")
    else:
        st.warning("❌ 'plakalar.txt' bulunamadı. Eşleşme yapılmayacak.")

    confirmed_ids = {}

    cap = cv2.VideoCapture(temp_video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2 = map(int, box[:4])
            score = box[4]
            if score < 0.3:
                continue
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], score, None))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            x1, y1, x2, y2 = int(l), int(t), int(w), int(h)
            plate_crop = frame[y1:y2, x1:x2]

            text = "UNDETECTED"
            try:
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ocr_results = reader.readtext(thresh)
                text = ocr_results[0][1] if ocr_results else "UNDETECTED"
                text = re.sub(r'[^A-Z0-9]', '', text.upper())
            except:
                pass

            if track_id in confirmed_ids:
                text = confirmed_ids[track_id]
                color = (0, 255, 0)
            elif text in valid_plates:
                confirmed_ids[track_id] = text
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    st.success("🎬 Video işleme tamamlandı.")
