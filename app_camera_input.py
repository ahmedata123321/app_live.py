import streamlit as st
import torch
import torchvision.transforms as transforms
from torch import nn
import timm
import gdown
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import base64
import os

st.set_page_config(page_title="Violence Detection Live", layout="wide")

# ---------------------------------------------------------
# إعدادات تحميل الموديل من Google Drive
# ---------------------------------------------------------
MODEL_PATH = "best_vit_lstm.pt"
MODEL_DRIVE_ID = "1GjmrQSLRtCwAtkk30ZOtFFXFqhOg6BxX"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# تعريف الموديل
# ---------------------------------------------------------
class ViT_LSTM_Classifier(nn.Module):
    def __init__(self, vit_name="vit_tiny_patch16_224", lstm_hidden=256, lstm_layers=1, num_classes=2, dropout=0.3):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        self.feat_dim = self.vit.num_features if hasattr(self.vit, "num_features") else 192
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.vit(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits

# تحميل الموديل
model = ViT_LSTM_Classifier().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ---------------------------------------------------------
# إعداد التحويل المسبق للفريمات
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------------------------------------------------------
# كود تشغيل الإنذار (صوت)
# ---------------------------------------------------------
AUDIO_FILE = "alert.wav"
alarm_base64 = ""
if os.path.exists(AUDIO_FILE):
    with open(AUDIO_FILE, "rb") as f:
        alarm_base64 = base64.b64encode(f.read()).decode()

audio_html = f"""
<audio id="alarm" src="data:audio/wav;base64,{alarm_base64}"></audio>
<script>
function playAlarm() {{
  var a = document.getElementById("alarm");
  if (a) {{
    a.currentTime = 0;
    a.play().catch(e=>console.log("play failed", e));
  }}
}}
</script>
"""
st.components.v1.html(audio_html, height=0)

# ---------------------------------------------------------
# WebRTC Video Transformer (تحليل مباشر)
# ---------------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frames_buffer = []
        self.seq_len = 8

    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_frame = transform(img_rgb)
        self.frames_buffer.append(tensor_frame)
        if len(self.frames_buffer) > self.seq_len:
            self.frames_buffer.pop(0)

        label = "Waiting..."
        color = (0, 255, 0)

        if len(self.frames_buffer) == self.seq_len:
            clip = torch.stack(self.frames_buffer).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(clip)
                pred = torch.argmax(out, dim=1).item()

            if pred == 1:
                label = "⚠️ Violent"
                color = (0, 0, 255)
                # تشغيل الإنذار على الواجهة
                st.components.v1.html("<script>playAlarm();</script>", height=0)
            else:
                label = "✅ Non-violent"
                color = (0, 255, 0)

            # عرض النتيجة على الفيديو
            cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------------------------------------
# واجهة Streamlit
# ---------------------------------------------------------
st.title("🚨 Real-time Violence Detection")
st.markdown("افتح الكاميرا مباشرة وسيظهر التصنيف في الوقت الفعلي (Violent / Non-violent).")

ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.success("✅ Ready — افتح الكاميرا بالسماح في المتصفح")
