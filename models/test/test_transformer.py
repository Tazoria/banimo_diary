import pandas as pd
from banimo_diary.models.utils.load_tokenizer import load_tokenizer

# def preprocessor_test():
# # 학습 데이터 전처리 테스트
# train_preprocessor = Preprocessor()
# # 추론 데이터 테스트
# infer_preprocessor = Preprocessor()
#
# print('=====processor(train)=====', train_preprocessor.tokenizer)
# print('=====processor(inference)=====', infer_preprocessor.tokenizer)
# print("=====train_input=====", train_input, sep='\n')
# print("=====train_output=====", train_output, sep='\n')
# print("=====infer_input=====", infer_input, sep='\n')
# print("=====infer_output_shape=====", infer_output_shape, sep='\n')



if __name__ == '__main__':
  train_data = pd.read_csv(r'/data/ChatbotData_merged.csv')
  train_data = train_data.dropna()
  q = train_data['Q']
  a = train_data['A']
  sentences = ['안녕하세요, 타조랑말입니다. 반갑습니다!', '오늘 공부가 잘 안됐어.']
  vocab_path = r'../vocab.txt'

  tokenizer = load_tokenizer(vocab_path)
  model_path = r'../save/weights/transformer_weight150.h5'

