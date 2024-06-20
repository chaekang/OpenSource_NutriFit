import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import requests
import io
import json
import base64
import global_list
from dotenv import load_dotenv
import os
import openai

# Load environment variables
load_dotenv()

# Load model
model_name = 'jhgan/ko-sroberta-multitask'
model = SentenceTransformer(model_name)

# Load the filtered DataFrame
LAST_DF_PATH = 'compact_kosroberta_recipes.pkl'
df = pd.read_pickle(LAST_DF_PATH)
df = df.reset_index(drop=True)

openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Streamlit
st.set_page_config(page_title="NutriFit", page_icon=":cook:", layout="wide")

# Initialize 'senior_mode' in session state
if 'senior_mode' not in st.session_state:
    st.session_state['senior_mode'] = False

# Define the msg_prompt dictionary
msg_prompt = {
    'recom': {
        'system': "You are a helpful assistant who recommend food items based on user question.",
        'user': "Write 1 sentence of a very simple greeting that starts with '추천드리겠습니다!' to recommend food items to users. and don't say any food name, say in korean",
    },
    'desc': {
        'system': "You are an assistant who very simply answers.",
        'user': "Please write a simple greeting starting with '요리에 대해 설명할게요' to explain the recipes to the user.",
    },
    'how': {
        'system': "You are a helpful assistant who kindly answers.",
        'user': "Please write a simple greeting starting with '방법을 말씀드릴게요' to explain the recipes to the user.",
    },
    'intent': {
        'system': "You are a helpful assistant who understands the intent of the user's query. and You answer in a short answer",
        'user': "Which category does the sentence below belong to: 'recommendation', 'description', how to cook'? pick one category. \n context:"
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

def toggle_senior_mode():
    st.session_state['senior_mode'] = not st.session_state['senior_mode']

st.button('시니어 모드 켜기' if not st.session_state['senior_mode'] else '시니어 모드 끄기', on_click=toggle_senior_mode)

if st.session_state['senior_mode']:
    st.markdown("""
        <style>
        body, h1, h2, h3, h4, h5, h6, p, span, div, input, button {
            font-size: 40px !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Main content
st.markdown("<span style='color:lightgray; font-style:italic; font-size:12px;'>24-1 오픈소스프로그래밍 기말 프로젝트 팀 NutriFit' </span>", unsafe_allow_html=True)
curr_dir = os.getcwd()
img_path = os.path.join(curr_dir, "NutriFit.jpg")
image = Image.open(img_path)
st.image(image)
img_path = os.path.join(curr_dir, "chat강록2-2.jpg")
image2 = Image.open(img_path)
st.image(image2)

st.write('\n')
st.write('\n\n')

chat_history = st.session_state.get("chat_history", [])

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

## 상위 5개 항목 출력(링크로 중복제거-민규님)
def get_query_sim_top_k(query, model, df):
    "쿼리와 데이터 간의 코사인 유사도를 측정하고 유사한 순위 5개 반환"
    if df['ko-sroberta-multitask-feature'].isnull().all():
        raise ValueError("DataFrame의 'ko-sroberta-multitask-feature' 열이 비어 있습니다.")
    
    query_encode = model.encode(query)
    cos_scores = util.pytorch_cos_sim(query_encode, df['ko-sroberta-multitask-feature'])[0]
    
    if cos_scores.size(0) == 0:
        raise ValueError("유사한 항목이 없습니다. 다른 쿼리를 시도해 주세요.")
    
    top_results = torch.topk(cos_scores, k=1)
    return top_results

# 챗봇 생성하기
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

query = None
with st.form(key='my_form'):
    query = st.text_input('입력창 ↓')
    submitted = st.form_submit_button('질문하기')

def user_interact(query, model, msg_prompt_init):
    global global_list
    
    # 사용자 메시지 기록이 비어있는지 확인하고 초기화합니다.
    if not hasattr(global_list, 'user_msg_history'):
        global_list.user_msg_history = []

    user_intent = set_prompt('intent', query, msg_prompt_init, None)
    user_intent = get_chatgpt_msg(user_intent).lower()
    print("user_intent : ", user_intent)
    
    intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
    intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
    print("intent_data_msg : ", intent_data_msg)
    
    try:
        if ('recom' in user_intent):
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
                raise ValueError("No previous user query available for description.")
            top_result = get_query_sim_top_k(global_list.user_msg_history[0]['content'][0], model, df)
            r_set_n = df.loc[top_result[1].numpy(), '요리']
            r_set_d = df.iloc[top_result[1].numpy(), :]['설명']
            r_set_d = json.loads(r_set_d.to_json(orient="records"))[0]
            global_list.user_msg_history.append({'role' : 'assistant', 'content' : r_set_d})
            return f' "{r_set_n.iloc[-1]}" 소개를 해드릴게요! \n\n {r_set_d}'
        
        elif 'how' in user_intent:
            if not global_list.user_msg_history:
                raise ValueError("No previous user query available for cooking instructions.")
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
    
    except ValueError as e:
        return str(e)

if __name__ == "__main__":
   
    chat_history = st.session_state.get("chat_history", [])
    
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if submitted and query:
        output = user_interact(query, model, msg_prompt)
        chat_history.append(query)
        st.session_state.past.append(query)
        st.session_state.past.append(output)
        
        if isinstance(output, tuple):
            st.markdown(f"<div style='padding-left: 70px;'> <h5> 🍳 {output[0]} </h5> </div>", unsafe_allow_html=True)
            st.markdown(output[1], unsafe_allow_html=True)
            st.markdown(output[2], unsafe_allow_html=True)
            st.markdown(output[4], unsafe_allow_html=True)
            st.markdown(f"<p style='padding-left: 70px; padding-right: 120px; font-size:16px; font-weight:bold; font-style:italic;'> (재료를 누르시면 구매페이지로 이동합니다.) <br> <span class='no-style'>{output[3]}</span> </p>", unsafe_allow_html=True)
            chat_history.append(output)
        else:
            st.write(output)
            chat_history.append(output)
        
        st.session_state["chat_history"] = chat_history
    
    if len(chat_history) > 2:
        for i in range(len(chat_history)-3, -1, -1):
            if i % 2 == 0:
                st.write(chat_history[i]) 
            else:
                if isinstance(chat_history[i], tuple):
                    st.markdown(f"<div style='padding-left: 70px;'> <h5> 🍳 {chat_history[i][0]} </h5> </div>", unsafe_allow_html=True)
                    st.markdown(chat_history[i][1], unsafe_allow_html=True)
                    st.markdown(chat_history[i][2], unsafe_allow_html=True)
                    st.markdown(chat_history[i][4], unsafe_allow_html=True)
                    st.markdown(f"<p style='padding-left: 70px; padding-right: 120px; font-size:16px; font-weight:bold; font-style:italic;'> (재료를 누르시면 구매페이지로 이동합니다.) <br> <span class='no-style'>{chat_history[i][3]}</span> </p>", unsafe_allow_html=True)
                else:
                    st.write(chat_history[i])
