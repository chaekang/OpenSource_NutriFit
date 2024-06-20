# 사진 옆 재료버튼 -> 링크연결 (챗봇형식0)
# 민규님 파일 연결
# 메세지 프롬프트 변경(강섭님)
# user_interact 함수 변경(강섭님)


# 데이터 분석
import pandas as pd
import numpy as np

# 진행시간 표시
# import swifter
from tqdm.notebook import tqdm

## 라이브러리 임포트
import streamlit as st 

# 파이토치
import torch

# 문장 임베딩, transformer 유틸리티
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import SentenceTransformer, models
# Owl-ViT를 위한 전처리, 객체 감지
from transformers import OwlViTProcessor, OwlViTForObjectDetection
# 파이프라인 구성
from transformers import pipeline
# GPT-2 토크나이저
from transformers import GPT2TokenizerFast

# 이미지 처리
from PIL import Image
# 사이킷 런
import sklearn.datasets as datasets
import sklearn.manifold as manifold

# 데이터 수집
import requests
from bs4 import BeautifulSoup

# 객체 복사
import copy
# JSON 형식 데이터 처리
import json
# 타입 힌트
from typing import List, Tuple, Dict

# 데이터베이스 활용
import sqlite3 
import pickle

# OpenAI API 활용
import openai 
import os # 운영체제
import sys # 파이썬 변수, 함수 엑세스 
# from dotenv import load_dotenv # 환경 변수 로드(API Key 보안)
import io

# 스트림릿 구현
import streamlit
from streamlit_chat import message
import global_list

from dotenv import load_dotenv # 환경 변수 로드(API Key 보안)


## 파일 및 API 가져오기
# app.py 파일이 위치한 경로 가져오기
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# app.py에서 만든 데이터프레임 불러오기
LAST_DF_PATH = os.path.join(APP_DIR, '../last_df.pkl')
df = pd.read_pickle(LAST_DF_PATH)
df = df.reset_index(drop=True)

load_dotenv()    
openai.api_key = os.getenv('OPENAI_API_KEY')
# openai.api_key = ''

# 실행 os 확인
cur_os = sys.platform

# 이전 대화 기록을 가져오기 위한 키
CHAT_HISTORY_KEY = "chat_history"

# 파생 변수
# - feature1 = '재료'
# - feature2 = '재료' + '요리'
# - feature3 = '재료' + '요리' + '종류'
# - feature4 = '재료' + '요리' + '종류' + '난이도'
# - feature5 = '재료' + '요리' + '종류' + '난이도' + '요리방법'
# - feature6 = '재료' + '요리' + '설명' + '종류' + '난이도' + '요리방법'



## 모델 선언
model_name = 'jhgan/ko-sroberta-multitask'
model = SentenceTransformer(model_name)



## 상위 5개 항목 출력(링크로 중복제거-민규님)
def get_query_sim_top_k(query, model, df):
    "쿼리와 데이터 간의 코사인 유사도를 측정하고 유사한 순위 5개 반환"
    query_encode = model.encode(query)
    cos_scores = util.pytorch_cos_sim(query_encode, df['ko-sroberta-multitask-feature'])[0]
    top_results = torch.topk(cos_scores, k=1)
    return top_results

# query = '고기 쪽파'
# top_result = get_query_sim_top_k(query, model, df)


# df.iloc[top_result[1].numpy(), :][['요리', '종류', '재료']]



## 메세지 프롬프트 생성
# intent에서 의도 파악하고 recom 혹은 desc 로 판단되는 것.
msg_prompt = {
    'recom' : {
                'system' : "You are a helpful assistant who recommend movie based on user question.", 
                'user' : "Write 1 sentence of a very simple greeting that starts with '추천드리겠습니다!' to recommend food items to users. and don't say any food name, say in korean", 
              },
    'desc' : {
                'system' : "You are a assistant who very simply answers.", 
                'user' : "Please write a simple greeting starting with '요리에 대해 설명할게요' to explain the recipes to the user.", 
              },
    'how' : {
                'system' : "You are a helpful assistant who kindly answers.", 
                'user' : "Please write a simple greeting starting with '방법을 말씀드릴게요' to explain the recipes to the user.", 
              },
    'intent' : {
                'system' : "You are a helpful assistant who understands the intent of the user's query. and You answer in a short answer",
                'user' : "Which category does the sentence below belong to: 'recommendation', 'description', how to cook'? pick one category. \n context:"
                }
}


## OpenAI API와 GPT-3 모델을 사용하여 msg에 대한 응답 생성
# 이전 대화내용을 고려하여 대화 생성.
def get_chatgpt_msg(msg):
    completion = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=msg
                    )
    return completion['choices'][0]['message']['content']



## intent와 사용자 쿼리를 바탕으로 prompt 생성
# 적절한 초기 메세지 생성, 사용자와의 자연스러운 대화구성 가능.
def set_prompt(intent, query, msg_prompt_init, model):
    '''prompt 형태를 만들어주는 함수'''
    m = dict()
    # 추천일 경우
    if 'recom' in intent:
        msg = msg_prompt_init['recom']  # 시스템 메세지를 가져옴
    # 설명일 경우
    elif 'desc' in intent:
        msg = msg_prompt_init['desc']  # 시스템 메세지를 가져옴
    # 요리방법일 경우
    elif 'how' in intent:
        msg = msg_prompt_init['how']  # 시스템 메세지를 가져옴
    # intent 파악
    else:
        msg = msg_prompt_init['intent']
        msg['user'] += f' {query} \n A:'
    for k, v in msg.items():
        m['role'], m['content'] = k, v
    return [m]



## 입력된 텍스트에 대해 gpt 모델을 사용하여 응답 생성.
# 함수 내부에서 입력된 텍스트를 토큰화하고, 생성된 응답을 디코딩 하여 반환.
# 입력 텍스트에 대해서만 모델을 사용하여 응답 생성(이전 대화 고려 x)
def generate_answer(model, tokenizer, input_text, max_len=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # generate text until the specified length
    output = model.generate(input_ids=input_ids, max_length=max_len, do_sample=True, top_p=0.92, top_k=50)
    # decode the generated text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

import base64
import webbrowser

def user_interact(query, model, msg_prompt_init):
    global global_list
    # 1. 사용자의 의도를 파악
    user_intent = set_prompt('intent', query, msg_prompt_init, None)
    user_intent = get_chatgpt_msg(user_intent).lower()
    print("user_intent : ", user_intent)
    
    # 2. 사용자의 쿼리에 따라 prompt 생성    
    intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
    intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
    print("intent_data_msg : ", intent_data_msg)
    
    # 3-1. 추천이면
    if 'recom' in user_intent:
        
        global_list.user_msg_history = []
        recom_msg = str()
        
        top_result = get_query_sim_top_k(query, model, df)
        top_index = top_result[1].numpy() if 'recom' in user_intent else top_result[1].numpy()[1:]
        
        r_set_d = df.iloc[top_index, :][['요리', '종류', '재료', '사진', '요리방법']]
        r_set_d = json.loads(r_set_d.to_json(orient="records"))
        for r in r_set_d:
            for _, v in r.items():
                recom_msg += f"{v} \n"
                img_url = r['사진']
                response = requests.get(img_url)
                image_bytes = io.BytesIO(response.content)
                image = Image.open(image_bytes)
                img_md = f"<img src='data:image/png;base64,{base64.b64encode(image_bytes.getvalue()).decode()}' style='padding-left: 70px; width:550px;'/>"

                r_ingredients = r['재료'].split()
                button_html = ''
                for ing in r_ingredients:
                    gs_url = f"https://m.gsfresh.com/shop/search/searchSect.gs?tq={ing}&mseq=S-11209-0301&keyword={ing}"
                    button_html += f"""<span style="white-space: nowrap;"><a href="{gs_url}" target="_blank" style="text-decoration: none; color: white; background-color: #008A7B; padding: 6px 12px; border-radius: 5px; margin-right: 5px; margin-bottom: 5px; margin-top: 5px;">{ing}</a></span>"""
                
                def recipe():
                    recipe_str = ''
                    for i, step in enumerate(recipe_steps):
                        step = step.strip()
                        if step:
                            recipe_str += f"{i+1}. {step}\n\n"
                    return recipe_str
                recipe_steps = r['요리방법'].replace('[', '').replace(']', '').replace("\\xa0", " ").replace("\\r\\n", ' ').split("', ")
                recipe_steps = [step.split("\\n") for step in recipe_steps]
                recipe_steps = [step for sublist in recipe_steps for step in sublist]
                recipe_steps = [step.strip() for step in recipe_steps]
                recipe_steps = [step.replace("'", "") for step in recipe_steps]
                
            recom_msg = f"추천메뉴 \" {r['요리']} \" ({r['종류']})"
            recipe_msg = ""
            recipe_msg += f"\"{r['요리']}\" 레시피를 알려드릴게요. \n\n "
            recipe_msg += recipe()

        global_list.user_msg_history.append({'role' : 'assistant', 'content' : [query, f"{intent_data_msg} {str(recom_msg)}"]})
        return recom_msg, img_md, f"{intent_data_msg}", button_html, recipe_msg, img_url
    
    elif 'desc' in user_intent:
        if not global_list.user_msg_history:
            return "오류: 이전 대화 내역이 없습니다."
        top_result = get_query_sim_top_k(global_list.user_msg_history[0]['content'][0], model, df)
        r_set_n = df.loc[top_result[1].numpy(), '요리']
        r_set_d = df.iloc[top_result[1].numpy(), :]['설명']
        r_set_d = json.loads(r_set_d.to_json(orient="records"))[0]
        global_list.user_msg_history.append({'role' : 'assistant', 'content' : r_set_d})
        return f' "{r_set_n.iloc[-1]}" 소개를 해드릴게요! \n\n {r_set_d}'

    elif 'how' in user_intent:
        if not global_list.user_msg_history:
            return "오류: 이전 대화 내역이 없습니다."
        top_result = get_query_sim_top_k(global_list.user_msg_history[0]['content'][0], model, df)
        r_set_d = df.iloc[top_result[1].numpy(), :]['요리방법']
        r_set_n = df.iloc[top_result[1].numpy(), :]['요리'].values[0]
        
        r_set_d_list = []
        for s in r_set_d:
            s_list = s.split("', ")
            for i in range(len(s_list)):
                s_list[i] = s_list[i].replace("'", "").replace(",", "").replace('[','').replace(']','').replace('\\xa0', ' ').replace('\\r\\n', '')
            r_set_d_list.extend(s_list)
            
        re_num = ""
        for i, s in enumerate(r_set_d_list, 1):
            re_num += f"{i}. {s} \n"
        global_list.user_msg_history.append({'role' : 'assistant', 'content' : r_set_d_list})

        return f'"{r_set_n}" 요리방법을 알려드릴게요! \n\n {re_num}'

            




st.set_page_config(page_title="Chat!강록", page_icon=":cook:", layout="wide")

if __name__ == "__main__":
    # 메인 구성하기
    st.markdown("<span style='color:lightgray; font-style:italic; font-size:12px;'>FINAL PROJECT(3조) '조이름은 최강록으로 하겠습니다. 그런데 이제 바질을 곁들인' </span>", 
                unsafe_allow_html=True)
        # 배너 이미지 넣기
    curr_dir = os.getcwd()
    img_path = os.path.join(curr_dir, "chat강록2-1.jpg")
    image = Image.open(img_path)
    st.image(image)
    img_path = os.path.join(curr_dir, "chat강록2-2.jpg")
    image2 = Image.open(img_path)
    st.image(image2)

    # st.markdown(':loudspeaker: <span style="color: #FF0033; font-weight: bold; font-size: 12px; font-style: italic;"> "고구마 맛탕 레피시 추천해줘." 혹은 "궁중떡볶이 설명해줘." 라고 입력해보세요!</span>', unsafe_allow_html=True)
    st.write('\n')
    st.write('\n\n')
   

    chat_history = st.session_state.get(CHAT_HISTORY_KEY, [])

    
    
    # 챗봇 생성하기
    if not hasattr(st.session_state, 'generated'):
        st.session_state.generated = []

    if not hasattr(st.session_state, 'past'):
        st.session_state.past = []


    


    # 쿼리 변형하기
    query = None
    with st.form(key='my_form'):
        query = st.text_input('입력창 ↓')
        submitted = st.form_submit_button('질문하기')




     # 수행문 만들기
    if submitted and query:
        output = user_interact(query, model, msg_prompt)
        chat_history.append(query)
        st.session_state.past.append(query)
        st.session_state.past.append(output)
        if isinstance(output, tuple): # 반환값이 튜플인 경우
            message(output[2],key=str(len(st.session_state.past)) + '2_assistant')
            st.markdown(f"<div style='padding-left: 70px;'> <h5> 🍳 {output[0]} </h5> </div>", unsafe_allow_html=True)
            st.markdown(output[1], unsafe_allow_html=True)
            # message(output[3], is_user=False, key=str(len(st.session_state.past)) + '_assistant')
            message(output[4],key=str(len(st.session_state.past)) + '4_assistant')
            st.markdown(f"<p style='padding-left: 70px; padding-right: 120px; font-size:16px; font-weight:bold; font-style:italic;'> (재료를 누르시면 구매페이지로 이동합니다.) <br> <span class='no-style'>{output[3]}</span> </p>", unsafe_allow_html=True)
            chat_history.append(output)
        else:
            message(output, is_user=False, key=str(len(st.session_state.past)) + '_assistant')
            chat_history.append(output)
        message(query, is_user=True, key=str(len(st.session_state.past)) + '_user')

        st.session_state[CHAT_HISTORY_KEY] = chat_history


        
    # # 출력하기
    # if len(chat_history) > 2:
    #     for i in range(len(chat_history)-3,-1,-1):
    #         if i % 2 == 0:
    #             message(chat_history[i], is_user=True, key=str(i)+ '_user+pass') 
    #         else : 
    #             message(chat_history[i], is_user=False, key=str(i)+ '_assistant+pass') 
    if len(chat_history) > 2:
        for i in range(len(chat_history)-3, -1, -1):
            if i % 2 == 0:
                message(chat_history[i], is_user=True, key=str(i)+ '_user+pass') 
            else :
                if isinstance(chat_history[i], tuple) :
                    message(chat_history[i][2], is_user=False, key=str(i) + '2_assistant+pass')
                    st.markdown(f"<div style='padding-left: 70px;'> <h5> 🍳 {chat_history[i][0]} </h5> </div>", unsafe_allow_html=True)
                    st.markdown(chat_history[i][1], unsafe_allow_html=True)
                    message(chat_history[i][4], is_user=False, key=str(i) + '4_assistant+pass')
                    st.markdown(f"<p style='padding-left: 70px; padding-right: 120px; font-size:16px; font-weight:bold; font-style:italic;'> (재료를 누르시면 구매페이지로 이동합니다.) <br> <span class='no-style'>{chat_history[i][3]}</span> </p>", unsafe_allow_html=True)
                else:
                    message(chat_history[i], is_user=False, key=str(i)+ '_assistant+pass')