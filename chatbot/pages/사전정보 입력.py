# 패키지 import
import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image
import requests
import torch
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import openai
import json

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load model
model_name = 'jhgan/ko-sroberta-multitask'
model = SentenceTransformer(model_name)

# Load recipe data
file_path = 'compact_kosroberta_recipes.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Initialize Streamlit
st.set_page_config(page_title="Chat!강록", page_icon=":cook:", layout="wide")

# 시니어 모드 상태 관리if 'senior_mode' not in st.session_state:
st.session_state.senior_mode = False

# 시니어 모드 버튼if st.button('시니어 모드'):
st.session_state.senior_mode = not st.session_state.senior_mode

# 텍스트 스타일 설정
text_style = "font-size: 20px;" if st.session_state.senior_mode else "font-size: 14px;"

# Main content
st.markdown(f"<span style='color:lightgray; font-style:italic; {text_style}'>FINAL PROJECT(3조) '조이름은 최강록으로 하겠습니다. 그런데 이제 바질을 곁들인' </span>", unsafe_allow_html=True)
curr_dir = os.getcwd()
img_path = os.path.join(curr_dir, "chat강록1-1.jpg")
image1 = Image.open(img_path)
st.image(image1)
img_path = os.path.join(curr_dir, "chat강록1-2.jpg")
image2 = Image.open(img_path)
st.image(image2)

st.markdown(f':loudspeaker: <span style="font-weight: bold; {text_style} font-style: italic;"> 현재 페이지는 사전정보 입력 페이지입니다.</span>', unsafe_allow_html=True)

# 알레르기 항목 리스트
allergies = {
    '우유': ['우유', '치즈', '버터', '크림', '요거트', '아이스크림'],
    '난류': ['계란','달걀', '메렌지', '마요네즈'],
    '땅콩': ['땅콩', '피넛버터', '땅콩크림', '땅콩깨'],
    '견과류': ['아몬드', '땅콩','호두', '피스타치오', '브라질너트', '마카다미아너트', '잣'],
    '대두': ['대두', '콩', '미소', '순두부', '된장', '콩나물', '콩물', '두부','간장'],
    '밀': ['밀가루', '밀떡', '면류', '케이크', '쿠키', '파스타', '빵', '시리얼'],
    '갑각류': ['새우', '랍스타', '게', '대게', '꽃게', '홍합', '조개류'],
    '조개류': [ '굴', '홍합', '전복', '조개','소라'],
    '생선': ['고등어', '연어', '참치', '멸치', '광어', '붕어', '오징어', '문어'],
    '육류': ['돼지고기', '햄', '소시지', '베이컨', '삼겹살', '쇠고기'],
    '복숭아': ['복숭아', '자두', '망고', '모과', '사과', '배', '포도']
}

# 건강 상태 리스트
health_conditions = ['비만', '당뇨']

# 성별 리스트
genders = ['남성', '여성']

# 사전 데이터프레임에 '건강상태' 열 추가 (예시 데이터 사용)# 실제 데이터에는 적절한 값을 추가해야 합니다.if '건강상태' not in data.columns:
data['건강상태'] = '일반'  # 기본값으로 '일반'을 추가

# 페이지 구성
st.write('\n')
st.write('\n')
with st.expander(f'###### Q1. 알레르기가 있으신가요?', expanded=True):
    st.markdown(f'<span style="color: blue; {text_style}"> Q1-1. 체크박스로 입력하기</span>', unsafe_allow_html=True)
    
    cols = st.columns(2)
    selected_allergies = []
    for i, allergy in enumerate(allergies):
        if i % 2 == 0:
            checkbox_col = cols[0]
        else:
            checkbox_col = cols[1]
        selected = checkbox_col.checkbox(allergy, key=allergy)
        if selected:
            selected_allergies.append(allergy)

    st.write("\n")
    st.write("\n")
    st.markdown(f'<span style="color: blue; {text_style}"> Q1-2. 직접 입력하기  ex) 복숭아, 수박 등</span>', unsafe_allow_html=True)
    other_input = st.text_input(' ', key='other_input')

    st.write('\n')
    st.write(f'###### ⬇️ 선택하신 알레르기 항목', unsafe_allow_html=True)
    selected_allergies = [allergy for allergy in allergies if st.session_state.get(allergy)]
    if len(selected_allergies) == 0 and not other_input:
        st.write('알레르기가 없습니다.', unsafe_allow_html=True)
    else:
        allergy_list = ", ".join(selected_allergies)
        if other_input:
            allergy_list += ", " + other_input
        st.write(allergy_list, unsafe_allow_html=True)

    if any(selected_allergies) or other_input:
        selected_allergies = [allergy for allergy in allergies if st.session_state.get(allergy)]
        other_allergy = other_input.strip()

        tmp = data.copy()
        for a in selected_allergies:
            tmp = tmp.loc[~tmp['재료'].str.contains('|'.join(allergies[a]))]
        df_al = tmp.copy()
        other_allergies = [x.strip() for x in other_allergy.split(',') if x.strip()]
        for allergy in other_allergies:
            df_al = df_al[~df_al['재료'].str.contains(allergy)]
    else:
        df_al = data

with st.expander(" 알레르기 정보 확인하기", expanded=True):
        st.markdown(f"<p style='color:red; {text_style}'> (일부 항목만 해당할 경우, 해당 항목을 직접 입력해주세요.)</p>", unsafe_allow_html=True)
        data = [
        ["체크 항목", "포함된 항목"],
        ['우유', '우유, 치즈, 버터, 크림, 요거트, 아이스크림'],
        ['난류', '계란, 달걀,     메렌지, 마요네즈'],
        ['땅콩', '땅콩, 피넛버터, 땅콩크림, 땅콩깨'],
        ['견과류', '아몬드, 땅콩, 호두, 피스타치오, 브라질너트, 마카다미아너트, 잣'],
        ['대두', '대두, 콩, 미소, 순두부, 된장, 콩나물, 콩물, 두부, 간장'],
        ['밀', '밀가루, 밀떡, 면류, 케이크, 쿠키, 파스타, 빵, 시리얼'],
        ['갑각류', '새우, 랍스타, 게, 대게, 꽃게, 홍합, 조개류'],
        ['생선', '고등어, 연어, 참치, 멸치, 광어, 붕어, 오징어, 문어'],
        ['육류', '돼지고기, 햄, 소시지, 베이컨, 삼겹살'],
        ['복숭아', '복숭아, 자두, 망고, 모과, 사과, 배, 포도']
    ]
        al_data = pd.DataFrame(data[1:], columns=data[0])
        st.write(al_data, unsafe_allow_html=True)

# 건강 상태 입력
with st.expander(f'###### Q2. 건강 상태를 선택해주세요.', expanded=True):
    selected_conditions = st.multiselect('건강 상태를 선택해주세요.', health_conditions)

# 연령 입력
with st.expander(f'###### Q3. 연령을 입력해주세요.', expanded=True):
    age = st.number_input('연령을 입력해주세요.', min_value=0, max_value=120, step=1)

# 성별 입력
with st.expander(f'###### Q4. 성별을 선택해주세요.', expanded=True):
    gender = st.selectbox('성별을 선택해주세요.', genders)

# 요리 범주 선택
st.write('\n')
menus = ['전체', '초대요리', '한식', '간식', '양식', '밑반찬', '채식', 
        '일식', '중식', '퓨전', '분식', '안주', '베이킹', '다이어트', 
        '도시락', '키토', '오븐 요리', '메인요리', '간단요리']

with st.expander(f'###### Q5. 원하는 요리 범주가 있으신가요?', expanded=True):
    cols = st.columns(4)
    selected_menus = []
    for i, menu in enumerate(menus):
        checkbox_col = cols[i % 4]
        selected = checkbox_col.checkbox(menu, key=menu)
        if selected:
            selected_menus.append(menu)

    if '전체' in selected_menus:
        df_me = df_al.copy()
    else:
        selected_menus = [menu for menu in selected_menus if menu != '전체']
        df_me = df_al[df_al['종류'].str.contains('|'.join(selected_menus))]

# 요리 난이도 선택
st.write('\n')
with st.expander(f'###### Q6. 원하는 요리 난이도가 있으신가요?', expanded=True):
    levels = st.multiselect('원하시는 난이도를 선택해주세요.', ['초보자', '중급자', '고급자'])

    if levels:
        filtered_df = df_me[df_me['난이도'].isin(['쉬움' if '초보자' in levels else 0,
                                                '보통' if '중급자' in levels else 0,
                                                '어려움' if '고급자' in levels else 0])]
    else:
        filtered_df = df_me

# 희망 요리시간 입력
st.write('\n')
with st.expander(f"###### Q7. 희망하는 요리시간이 있으신가요?", expanded=True):
    time = st.text_input('희망하는 최대 소요시간을 입력해주세요. ex) 120 (분 단위 숫자로 입력)')
    last_df = filtered_df.copy()

    if time:
        time = int(time)
        last_df = last_df[last_df['소요시간'] <= time]

# 건강 상태에 따른 필터링
if '비만' in selected_conditions:
    last_df = last_df[last_df['질병'].str.contains('비만')]
if '당뇨' in selected_conditions:
    last_df = last_df[last_df['질병'].str.contains('당뇨')]

st.write('\n')
st.write('\n')
if st.button(label='저장'):
    with open('last_df.pkl', 'wb') as f:
        pickle.dump(last_df, f)
    st.write('저장되었습니다.')

st.markdown(f'<span style="color: red; font-weight: bold; {text_style} font-style: italic;"> "저장" 버튼을 눌러야 정보가 저장됩니다.</span>', unsafe_allow_html=True)

# ChatGPT 메시지 프롬프트
msg_prompt = {
    'recom' : {
        'system' : "You are a helpful assistant who recommend food items based on user question.", 
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

# OpenAI API와 GPT-3 모델을 사용하여 msg에 대한 응답 생성
def get_chatgpt_msg(msg):
    completion = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=msg
                    )
    return completion['choices'][0]['message']['content']

# intent와 사용자 쿼리를 바탕으로 prompt 생성
def set_prompt(intent, query, msg_prompt_init):
    m = dict()
    if 'recom' in intent:
        msg = msg_prompt_init['recom']
    elif 'desc' in intent:
        msg = msg_prompt_init['desc']
    elif 'how' in intent:
        msg = msg_prompt_init['how']
    else:
        msg = msg_prompt_init['intent']
        msg['user'] += f' {query} \n A:'
    for k, v in msg.items():
        m['role'], m['content'] = k, v
    return [m]

# 입력된 텍스트에 대해 gpt 모델을 사용하여 응답 생성
def generate_answer(model, tokenizer, input_text, max_len=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=max_len, do_sample=True, top_p=0.92, top_k=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def get_query_sim_top_k(query, model, df):
    query_encode = model.encode(query)
    cos_scores = util.pytorch_cos_sim(query_encode, df['feature'])[0]
    top_results = torch.topk(cos_scores, k=5)
    return top_results

def user_interact(query, model, msg_prompt_init):
    user_intent = set_prompt('intent', query, msg_prompt_init)
    user_intent = get_chatgpt_msg(user_intent).lower()
    
    intent_data = set_prompt(user_intent, query, msg_prompt_init)
    intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
    
    if ('recom' in user_intent):
        recom_msg = str()
        top_result = get_query_sim_top_k(query, model, data)
        top_index = top_result[1].numpy() if 'recom' in user_intent else top_result[1].numpy()[1:]
        r_set_d = data.iloc[top_index, :][['요리', '종류', '재료', '사진', '요리방법']]
        r_set_d = json.loads(r_set_d.to_json(orient="records"))
        for r in r_set_d:
            recom_msg += f"{r['요리']} ({r['종류']}) \n{r['재료']} \n"
        return recom_msg
    
    elif 'desc' in user_intent:
        top_result = get_query_sim_top_k(global_list.user_msg_history[0]['content'][0], model, data)
        r_set_n = data.loc[top_result[1].numpy(), '요리']
        r_set_d = data.iloc[top_result[1].numpy(), :]['설명']
        r_set_d = json.loads(r_set_d.to_json(orient="records"))[0]
        return f' "{r_set_n.iloc[-1]}" 소개를 해드릴게요! \n\n {r_set_d}'
    
    elif 'how' in user_intent:
        top_result = get_query_sim_top_k(global_list.user_msg_history[0]['content'][0], model, data)
        r_set_d = data.iloc[top_result[1].numpy(), :]['요리방법']
        r_set_n = data.iloc[top_result[1].numpy(), :]['요리'].values[0]
        r_set_d_list = []
        for s in r_set_d:
            s_list = s.split("', ")
            for i in range(len(s_list)):
                s_list[i] = s_list[i].replace("'", "").replace(",", "").replace('[','').replace(']','').replace('\\xa0', ' ').replace('\\r\\n', '')
            r_set_d_list.extend(s_list)
        re_num = ""
        for i, s in enumerate(r_set_d_list, 1):
            re_num += f"{i}. {s} \n"
        return f'"{r_set_n}" 요리방법을 알려드릴게요! \n\n {re_num}'

if __name__ == "__main__":
    st.markdown(f"<span style='color:lightgray; font-style:italic; {text_style}'>FINAL PROJECT(3조) '조이름은 최강록으로 하겠습니다. 그런데 이제 바질을 곁들인' </span>", unsafe_allow_html=True)
    curr_dir = os.getcwd()
    img_path = os.path.join(curr_dir, "chat강록2-1.jpg")
    image = Image.open(img_path)
    st.image(image)
    img_path = os.path.join(curr_dir, "chat강록2-2.jpg")
    image2 = Image.open(img_path)
    st.image(image2)

    # 챗봇 초기화
    CHAT_HISTORY_KEY = "chat_history"
    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []

    chat_history = st.session_state[CHAT_HISTORY_KEY]

    if not hasattr(st.session_state, 'generated'):
        st.session_state.generated = []

    if not hasattr(st.session_state, 'past'):
        st.session_state.past = []

    query = None
    with st.form(key='my_form'):
        query = st.text_input('입력창 ↓')
        submitted = st.form_submit_button('질문하기')

    if submitted and query:
        output = user_interact(query, model, msg_prompt)
        chat_history.append(query)
        st.session_state.past.append(query)
        st.session_state.past.append(output)
        if isinstance(output, tuple):
            st.markdown(f"<div style='padding-left: 70px;'> <h5> 🍳 {output[0]} </h5> </div>", unsafe_allow_html=True)
            st.markdown(output[1], unsafe_allow_html=True)
            st.markdown(f"<p style='padding-left: 70px; padding-right: 120px; font-size:16px; font-weight:bold; font-style:italic;'> (재료를 누르시면 구매페이지로 이동합니다.) <br> <span class='no-style'>{output[3]}</span> </p>", unsafe_allow_html=True)
            chat_history.append(output)
        else:
            st.write(output)
            chat_history.append(output)
        st.session_state[CHAT_HISTORY_KEY] = chat_history

    if len(chat_history) > 2:
        for i in range(len(chat_history) - 3, -1, -1):
            if i % 2 == 0:
                st.write(chat_history[i]) 
            else:
                if isinstance(chat_history[i], tuple):
                    st.markdown(f"<div style='padding-left: 70px;'> <h5> 🍳 {chat_history[i][0]} </h5> </div>", unsafe_allow_html=True)
                    st.markdown(chat_history[i][1], unsafe_allow_html=True)
                    st.markdown(f"<p style='padding-left: 70px; padding-right: 120px; font-size:16px; font-weight:bold; font-style:italic;'> (재료를 누르시면 구매페이지로 이동합니다.) <br> <span class='no-style'>{chat_history[i][3]}</span> </p>", unsafe_allow_html=True)
                else:
                    st.write(chat_history[i])

