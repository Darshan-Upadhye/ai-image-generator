import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🖼️ Text-to-Image Generator using Stable Diffusion")
st.markdown("Enter a prompt and generate an image using Stable Diffusion!")

# Load model
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()

# Prompt input
prompt = st.text_input("Enter your prompt:", value="A futuristic city in sunset", max_chars=100)

if st.button("Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)

            # Allow image download
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            st.download_button(
                label="📥 Download Image",
                data=buffered.getvalue(),
                file_name="generated_image.png",
                mime="image/png"
            )

# footer using HTML + CSS
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            padding: 10px 0;
            background-color: #f9f9f9;
            text-align: center;
            font-size: 0.9rem;
            color: #555;
            border-top: 1px solid #eaeaea;
        }

        .footer a {
            text-decoration: none;
            color: #007acc;
        }

        .footer a:hover {
            text-decoration: underline;
        }
    </style>

    <div class="footer">
        © 2025 Developed by <a href="https://www.linkedin.com/in/darshan-upadhye-02a9a5287?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3Bh1sFDQTwSO6Zc0SEZPz2jw%3D%3D" target="_blank">Darshan Akshay Upadhye</a>
    </div>
""", unsafe_allow_html=True)
