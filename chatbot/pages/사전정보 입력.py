import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load model
model_name = 'jhgan/ko-sroberta-multitask'
model = SentenceTransformer(model_name)

# Load recipe data
file_path = 'compact_kosroberta_recipes.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Initialize Streamlit
st.set_page_config(page_title="NutriFit", page_icon=":cook:", layout="wide")

# Button to enable senior mode
if 'senior_mode' not in st.session_state:
    st.session_state['senior_mode'] = False

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
image1 = Image.open(img_path)
st.image(image1)
img_path = os.path.join(curr_dir, "chat강록1-2.jpg")
image2 = Image.open(img_path)
st.image(image2)

st.markdown(':loudspeaker: <span style="font-weight: bold; font-size: 14px; font-style: italic;"> 현재 페이지는 사전정보 입력 페이지입니다.</span>', unsafe_allow_html=True)

# 알레르기 항목 리스트
allergies = {
    '우유': ['우유', '치즈', '버터', '크림', '요거트', '아이스크림'],
    '난류': ['계란', '달걀', '메렌지', '마요네즈'],
    '땅콩': ['땅콩', '피넛버터', '땅콩크림', '땅콩깨'],
    '견과류': ['아몬드', '땅콩', '호두', '피스타치오', '브라질너트', '마카다미아너트', '잣'],
    '대두': ['대두', '콩', '미소', '순두부', '된장', '콩나물', '콩물', '두부', '간장'],
    '밀': ['밀가루', '밀떡', '면류', '케이크', '쿠키', '파스타', '빵', '시리얼'],
    '갑각류': ['새우', '랍스타', '게', '대게', '꽃게', '홍합', '조개류'],
    '조개류': ['굴', '홍합', '전복', '조개', '소라'],
    '생선': ['고등어', '연어', '참치', '멸치', '광어', '붕어', '오징어', '문어'],
    '육류': ['돼지고기', '햄', '소시지', '베이컨', '삼겹살', '쇠고기'],
    '복숭아': ['복숭아', '자두', '망고', '모과', '사과', '배', '포도']
}

# 건강 상태 리스트
health_conditions = ['비만', '당뇨']

# 성별 리스트
genders = ['남성', '여성']

# 사전 데이터프레임에 '건강상태' 열 추가 (예시 데이터 사용)# 실제 데이터에는 적절한 값을 추가해야 합니다.
if '질병' not in data.columns:
    data['질병'] = '일반'  # 기본값으로 '일반'을 추가

# 페이지 구성
st.write('\n')
st.write('\n')
with st.expander(f'###### Q1. 알레르기가 있으신가요?', expanded=True):
    st.markdown('<span style="color: blue;"> Q1-1. 체크박스로 입력하기</span>', unsafe_allow_html=True)

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
    st.markdown('<span style="color: blue;"> Q1-2. 직접 입력하기  ex) 복숭아, 수박 등</span>', unsafe_allow_html=True)
    other_input = st.text_input(' ', key='other_input')

    st.write('\n')
    st.write('###### ⬇️ 선택하신 알레르기 항목')
    selected_allergies = [allergy for allergy in allergies if st.session_state.get(allergy)]
    if len(selected_allergies) == 0 and not other_input:
        st.write('알레르기가 없습니다.')
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

with st.expander(" 알레르기 정보 확인하기"):
    st.markdown("<p style='color:red'> (일부 항목만 해당할 경우, 해당 항목을 직접 입력해주세요.)</p>", unsafe_allow_html=True)
    data_info = [
        ["체크 항목", "포함된 항목"],
        ['우유', '우유, 치즈, 버터, 크림, 요거트, 아이스크림'],
        ['난류', '계란, 달걀, 메렌지, 마요네즈'],
        ['땅콩', '땅콩, 피넛버터, 땅콩크림, 땅콩깨'],
        ['견과류', '아몬드, 땅콩, 호두, 피스타치오, 브라질너트, 마카다미아너트, 잣'],
        ['대두', '대두, 콩, 미소, 순두부, 된장, 콩나물, 콩물, 두부, 간장'],
        ['밀', '밀가루, 밀떡, 면류, 케이크, 쿠키, 파스타, 빵, 시리얼'],
        ['갑각류', '새우, 랍스타, 게, 대게, 꽃게, 홍합, 조개류'],
        ['생선', '고등어, 연어, 참치, 멸치, 광어, 붕어, 오징어, 문어'],
        ['육류', '돼지고기, 햄, 소시지, 베이컨, 삼겹살'],
        ['복숭아', '복숭아, 자두, 망고, 모과, 사과, 배, 포도']
    ]
    al_data = pd.DataFrame(data_info[1:], columns=data_info[0])
    st.write(al_data, unsafe_allow_html=True)

# 건강 상태 입력
with st.expander('###### Q2. 건강 상태를 선택해주세요.'):
    selected_conditions = st.multiselect('건강 상태를 선택해주세요.', health_conditions)

# 연령 입력
with st.expander('###### Q3. 연령을 입력해주세요.'):
    age = st.number_input('연령을 입력해주세요.', min_value=0, max_value=120, step=1)

# 성별 입력
with st.expander('###### Q4. 성별을 선택해주세요.'):
    gender = st.selectbox('성별을 선택해주세요.', genders)
# 요리 범주 선택
st.write('\n')
menus = ['전체', '초대요리', '한식', '간식', '양식', '밑반찬', '채식', 
        '일식', '중식', '퓨전', '분식', '안주', '베이킹', '다이어트', 
        '도시락', '키토', '오븐 요리', '메인요리', '간단요리']

with st.expander('###### Q5. 원하는 요리 범주가 있으신가요?'):
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
with st.expander('###### Q6. 원하는 요리 난이도가 있으신가요?'):
    levels = st.multiselect('원하시는 난이도를 선택해주세요.', ['초보자', '중급자', '고급자'])

    if levels:
        filtered_df = df_me[df_me['난이도'].isin(['쉬움' if '초보자' in levels else 0,
                                                '보통' if '중급자' in levels else 0,
                                                '어려움' if '고급자' in levels else 0])]
    else:
        filtered_df = df_me

# 희망 요리시간 입력
st.write('\n')
with st.expander("###### Q7. 희망하는 요리시간이 있으신가요?"):
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

st.markdown("<p style='color:red'> (일부 항목만 해당할 경우, 해당 항목을 직접 입력해주세요.)</p>", unsafe_allow_html=True)
