import google.generativeai as genai
from PIL import Image 
import streamlit as st
import io

st.title("Solve Math From Image")

uploaded_files = st.file_uploader("Math Image Upload")
if uploaded_files is not None:
    genai.configure(api_key="AIzaSyDxPX2Bq9q4kf15723PlvcVG1h3iYpEzeQ")
    # Choose a Gemini model.
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # # Upload the file and print a confirmation.
    # sample_file = genai.upload_file(path="mathproblem.jpg",
    #                             display_name="Jetpack drawing")
    imageFile = io.BytesIO(uploaded_files.read())
    sample_file = Image.open(imageFile)
    # print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
        

    # Prompt the model with text and the previously uploaded image.
    response = model.generate_content([sample_file, "Solve this math"])
    # print(">" + response.text)
    col1, col2 = st.columns(2,gap='large')
    with col1:
        st.title("Question")
        st.image(uploaded_files)
    with col2:
        st.title("Solution")
        st.markdown(response.text)