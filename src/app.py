import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import sys
import os
import tempfile

sys.path.append(os.path.dirname(__file__))
from inference.predict import load_model, predict_mask
from inference.postprocess import draw_lane_curves, compute_lane_offset

st.set_page_config(page_title="Lane Detection", layout="wide")

st.title("🛣️ Autonomous Lane Detection")
st.write("Detect lane lines using a U-Net + MobileNetV2 model trained on TuSimple.")


@st.cache_resource
def get_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('models/best_model_v2_safe.pth', device=device)
    return model, device


model, device = get_model()

tab1, tab2 = st.tabs(["📷 Image", "🎥 Video"])

# ---------------- IMAGE TAB ----------------
with tab1:
    uploaded_file = st.file_uploader("Upload a road image", type=['jpg', 'jpeg', 'png'], key="img_upload")

    image_source = None
    if uploaded_file is not None:
        image_source = uploaded_file
    elif 'selected_sample' in st.session_state:
        image_source = st.session_state['selected_sample']

    if image_source is not None:
        if st.button("🔄 Try another image"):
            if 'selected_sample' in st.session_state:
                del st.session_state['selected_sample']
            st.rerun()

        image_size = (256, 512)
        original_image = Image.open(image_source).convert('RGB').resize(image_size)

        img_array = np.array(original_image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).unsqueeze(0).to(device)

        with st.spinner("Detecting lanes..."):
            mask = predict_mask(model, img_tensor)
            curved_image, fitted_curves = draw_lane_curves(original_image, mask)
            offset = compute_lane_offset(fitted_curves, mask.shape[1], mask.shape[0])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(original_image, width=400)
        with col2:
            st.subheader("Detected Lanes")
            st.image(curved_image, width=400)

        if offset is not None:
            st.metric("Estimated vehicle offset from lane center", f"{offset:.1f} px")
        else:
            st.warning("Could not estimate offset (fewer than 2 lanes detected)")

    else:
        st.info("👆 Upload an image to get started")
        st.write("Or try one of the sample images:")

        sample_dir = 'data/sample_test_images'
        if os.path.exists(sample_dir):
            samples = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
            cols = st.columns(len(samples))
            for i, sample in enumerate(samples):
                with cols[i]:
                    st.image(os.path.join(sample_dir, sample), caption=sample, use_container_width=True)
                    if st.button("Use this image", key=f"sample_{i}"):
                        st.session_state['selected_sample'] = os.path.join(sample_dir, sample)
                        st.rerun()

# ---------------- VIDEO TAB ----------------
with tab2:
    st.write("Upload a short video (under ~15 seconds recommended for reasonable processing time).")
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'], key="video_upload")

    if uploaded_video is not None:
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(uploaded_video.read())
        input_path = tfile_in.name

        st.info("Processing video frame-by-frame — this can take a few minutes depending on video length and your hardware.")

        output_path = input_path.replace('.mp4', '_annotated.mp4')

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        image_size = (256, 512)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

        progress_bar = st.progress(0, text="Processing video...")

        frame_skip = 2  # process every 2nd frame, reuse annotation for skipped ones
        last_curved_bgr = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0 or last_curved_bgr is None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb).resize(image_size)

                img_array = np.array(pil_image, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).unsqueeze(0).to(device)

                mask = predict_mask(model, img_tensor)
                curved_image, fitted_curves = draw_lane_curves(pil_image, mask)
                offset = compute_lane_offset(fitted_curves, mask.shape[1], mask.shape[0])

                curved_image_resized = cv2.resize(np.array(curved_image), (orig_width, orig_height))
                curved_bgr = cv2.cvtColor(curved_image_resized, cv2.COLOR_RGB2BGR)

                if offset is not None:
                    cv2.putText(curved_bgr, f"Offset: {offset:.1f}px", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                last_curved_bgr = curved_bgr.copy()
            else:
                curved_bgr = last_curved_bgr

            out.write(curved_bgr)

            frame_idx += 1
            progress_bar.progress(min(frame_idx / frame_count, 1.0), text=f"Processing frame {frame_idx}/{frame_count}")

        cap.release()
        out.release()

        # Re-encode to browser-compatible H.264 so playback actually works
        h264_output_path = output_path.replace('.mp4', '_h264.mp4')
        os.system(f'ffmpeg -y -i {output_path} -vcodec libx264 -pix_fmt yuv420p {h264_output_path} -loglevel quiet')

        st.success("Done! Here's your annotated video:")
        st.video(h264_output_path)

        os.unlink(input_path)
        os.unlink(output_path)
    else:
        st.info("👆 Upload a video to get started")