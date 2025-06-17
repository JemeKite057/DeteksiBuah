import streamlit as st

# Peringatan awal jika YOLOv5 atau dependensi tidak ditemukan
try:
    import torch
    from PIL import Image
except ModuleNotFoundError as e:
    st.error("Beberapa modul belum terinstal. Pastikan sudah menjalankan:")
    st.code("pip install torch torchvision pillow")
    st.stop()

import os
import time
import numpy as np

st.set_page_config(page_title="Deteksi Buah Segar/Busuk dengan YOLOv5", layout="centered")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Peringatan: File CSS '{file_name}' tidak ditemukan. Menggunakan gaya default.")
    except Exception as e:
        st.warning(f"Peringatan: Gagal memuat file CSS. Error: {e}")

load_css('style.css')

@st.cache_resource
def load_yolov5_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='fresh_rotten_yolov5.pt', force_reload=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLOv5: {e}")
        st.stop()

model_yolo = load_yolov5_model()

st.title("Deteksi Buah: Segar atau Busuk? (YOLOv5)")
st.markdown("---")

st.write("Unggah gambar buah Apel, Pisang, atau Jeruk untuk mendeteksi dan mengklasifikasikan apakah buah tersebut segar atau busuk.")

uploaded_file = st.file_uploader("Pilih gambar buah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Gambar yang Diunggah', use_container_width=True)
    st.write("")

    with st.spinner('Menganalisis gambar dengan YOLOv5...'):
        time.sleep(1.0)
        img = Image.open(uploaded_file)
        results = model_yolo(img)

        st.subheader("📊 Hasil Deteksi")
        results.render()
        st.image(results.ims[0], caption='Hasil Deteksi', use_container_width=True)

        # Menampilkan label dan confidence setiap deteksi
        for *box, conf, cls in results.xyxy[0].tolist():
            label = results.names[int(cls)]
            st.write(f"- **{label}** dengan keyakinan **{conf * 100:.2f}%**")

st.markdown("---")
st.markdown("""
<div class="footer">
    Aplikasi Deteksi Buah oleh AkmalAditAlbarr | Menggunakan Streamlit dan YOLOv5
</div>
""", unsafe_allow_html=True)
