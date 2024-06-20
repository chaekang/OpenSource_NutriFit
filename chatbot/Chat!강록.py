import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="NutriFit", page_icon=":cook:", layout="wide")

# 메인 구성하기
st.markdown("<span style='color:lightgray; font-style:italic; font-size:12px;'>24-1 오픈소스프로그래밍 기말 프로젝트 팀 NutriFit' </span>", 
            unsafe_allow_html=True)
    # 배너 이미지 넣기
curr_dir = os.getcwd()
img_path = os.path.join(curr_dir, "NutriFit.jpg")
image1 = Image.open(img_path)
st.image(image1)
st.write('\n')
img_path = os.path.join(curr_dir, "notion.jpg")
image2 = Image.open(img_path)
st.image(image2)