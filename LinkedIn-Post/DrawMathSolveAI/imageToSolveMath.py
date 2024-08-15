import google.generativeai as genai
from PIL import Image 
import streamlit as st
import io

st.set_page_config(layout="wide")
st.image("Math1.png",use_column_width=True)

st.title("Solve Math Problems from an Image")
# video_file = open('Math.mp4', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes,start_time=0)
uploaded_files = st.file_uploader("Math Image Upload")
if uploaded_files is not None:
    genai.configure(api_key="AIzaSyDxPX2Bq9q4kf15723PlvcVG1h3iYpEzeQ")
    # Choose a Gemini model.
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    imageFile = io.BytesIO(uploaded_files.read())
    sample_file = Image.open(imageFile)
    # Prompt the model with text and the previously uploaded image.
    response = model.generate_content([sample_file, "Solve this math"])
    col1, col2 = st.columns(2,gap='large')
    with col1:
        st.title("Question")
        st.image(uploaded_files)
    with col2:
        st.title("Solution")
        
        st.markdown(response.text)
        