import json
import pandas as pd


def get_data(json_file):
    # 필요한 8개의 데이터 컬럼을 만들기 위한 변수 선언
    situation, disease, age, gender, question, answer, turn, label = [], [], [], [], [], [], [], []
    for sample in json_file:
        # emotion-id로부터 situation, disease, emotion 분리
        # emotion-id: Situation-Disease-Emotion의 형태
        emotion_id = sample['profile']['emotion']['emotion-id']
        S = emotion_id.split('_')[0]
        D = emotion_id.split('_')[1]
        E = emotion_id.split('_')[2]

        # persona_id로부터 age, gender 분리
        # persona_id: Age-Gender-Computer(응답, 사용X)의 형태
        persona_id = sample['profile']['persona']['persona-id']
        A = persona_id.split('_')[0]
        G = persona_id.split('_')[1]

        # content로부터 사람대화, 시스템 응답을 가져옴
        # content: 사람대화(HS01, HS02, HS03), 시스템 응답(SS01, SS02, SS03)
        # turn: 대화 턴 순번
        content = sample['talk']['content']
        # 사람 대화 - 비어있는 경우도 있음(주로 3번째 대화)
        if content['HS01'] not in ['', None]:
            situation.append(S)
            disease.append(D)
            age.append(A)
            gender.append(G)
            question.append(content['HS01'])
            answer.append(content['SS01'])
            turn.extend([1])  # 1 for HS01, 1 for SS01
            label.append(E)

        if content['HS02'] not in ['', None]:
            situation.append(S)
            disease.append(D)
            age.append(A)
            gender.append(G)
            question.append(content['HS02'])
            answer.append(content['SS02'])
            turn.extend([2])
            label.append(E)

        if content['HS03'] not in ['', None]:
            situation.append(S)
            disease.append(D)
            age.append(A)
            gender.append(G)
            question.append(content['HS03'])
            answer.append(content['SS03'])
            turn.extend([3])
            label.append(E)

    # 데이터프레임 구성
    df = pd.DataFrame({
        'situation': situation,
        'disease': disease,
        'age': age,
        'gender': gender,
        'Q': question,
        'A': answer,
        'turn': turn,
        'label': label,
    })

    df = df.dropna()
    return df


def make_clear(df):
    # situation 대치
    df_cleaned = df.replace({
        # situation 대치
        'situation': {
            'S01': '가족관계',
            'S02': '학업-진로',
            'S03': '학교폭력-따돌림',
            'S04': '대인관계',
            'S05': '연애-결혼-출산',
            'S06': '진로-취업-직장',
            'S07': '대인관계(부부-자녀)',
            'S08': '재정-은퇴-노후준비',
            'S09': '건강',
            'S10': '직장-업무스트레스',
            'S11': '건강-죽음',
            'S12': '대인관계(노년)',
            'S13': '재정'
        },
        # disease(만성질병) 대치
        'disease': {
            'D01': '있음',
            'D02': '없음'
        },
        # 나이 대치
        'age': {
            'A01': '청소년',
            'A02': '청년',
            'A03': '중년',
            'A04': '노년'
        },
        # 성별 대치
        'gender': {
            'G01': '남자',
            'G02': '여자'
        }
    })

    # 60개의 세분화된 감정을 6개의 대분류로 통합
    # angry(E10~E19), sad(E20~E29), insecure(E30~E39), broken_heart(E40~E49), embarrassed(E50~E59), happy(E60~E69)

    new_emotion = []
    for e in df_cleaned['label']:
        if e[1] == '1':
            new_emotion.append('angry')
        elif e[1] == '2':
            new_emotion.append('sad')
        elif e[1] == '3':
            new_emotion.append('insecure')
        elif e[1] == '4':
            new_emotion.append('broken_heart')
        elif e[1] == '5':
            new_emotion.append('embarrassed')
        elif e[1] == '6':
            new_emotion.append('happy')
        else:
            print('ValueError: emotion')

    df_cleaned['label'] = new_emotion
    df_cleaned = df_cleaned.dropna()
    return df_cleaned


if __name__ == '__main__':
    # JSON파일을 읽어와 필요한 데이터들을 데이터프레임으로 구성
    with open('../data/sentiment_training.json', encoding='utf-8') as f:
        train_js = json.loads(f.read())
    with open('../data/sentiment_validation.json', encoding='utf-8') as f:
        valid_js = json.loads(f.read())

    chatbot_data = pd.read_csv('../data/ChatbotData.csv', encoding='utf-8')

    # 데이터를 보기 편하게 바꾸기
    # train 데이터
    train = get_data(train_js)
    train_clear = make_clear(train)

    # valid 데이터
    valid = get_data(valid_js)
    valid_clear = make_clear(valid)

    # 데이터를 source와 target으로 파일을 분리해 따로 구성할 때
    train_src = train_clear['Q']
    train_tgt = train_clear['A']
    valid_src = valid_clear['Q']
    valid_tgt = valid_clear['A']

    train_clear.to_csv('../data/clear_data.csv', index=False)
    valid_clear.to_csv('../data/clear_data.csv', index=False)
    train_src.to_csv('../data/train_src.txt', sep='\n', header=False, index=False, encoding='utf-8')
    train_tgt.to_csv('../data/train_tgt.txt', sep='\n', header=False, index=False, encoding='utf-8')
    valid_src.to_csv('../data/valid_src.txt', sep='\n', header=False, index=False, encoding='utf-8')
    valid_tgt.to_csv('../data/valid_tgt.txt', sep='\n', header=False, index=False, encoding='utf-8')

    chatbot_data = pd.concat([chatbot_data, train_clear[['Q', 'A', 'label']]])
    chatbot_data.to_csv('./data/ChatbotData_merged.csv', index=False, encoding='utf-8')

