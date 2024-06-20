import streamlit as st
from PIL import Image
import os


st.set_page_config(page_title="Chat!강록", page_icon=":cook:", layout="wide")

# 메인 구성하기
st.markdown("<span style='color:lightgray; font-style:italic; font-size:12px;'>FINAL PROJECT(3조) '조이름은 최강록으로 하겠습니다. 그런데 이제 바질을 곁들인' </span>", 
            unsafe_allow_html=True)
    # 배너 이미지 넣기
curr_dir = os.getcwd()
img_path = os.path.join(curr_dir, "main.jpg")
image1 = Image.open(img_path)
st.image(image1)
st.write('\n')
img_path = os.path.join(curr_dir, "notion.jpg")
image2 = Image.open(img_path)
st.image(image2)