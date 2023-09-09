import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  '''
  [ get_angles ]
  - Positional Encoding 행렬의 각 요소에 대한 가중치를 계산
  - position
    - 시퀀스 내에서 토큰의 위치
    - 서로 다른 위치의 요소에 대한 고유한 위치정보
    - 1부터 시작
  - i 
    - Positional Encoding 행렬 내에서 각 요소의 인덱스
    - 각 Positional Encoding 행렬 요소에 대한 고유한 인덱스
    - 0~N
  - 각 가중치는 position, i에 따라 다르게 설정됨
  '''
  def get_angles(self, position, i, d_model):
    angles = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
      position=torch.arange(position, dtype=torch.float32).unsqueeze(1),
      i=torch.arange(d_model, dtype=torch.float32).unsqueeze(0),
      d_model=d_model
    )

    # 짝수 인덱스(2i)에는 사인 함수 적용
    sines = torch.sin(angle_rads[:, 0::2])

    # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
    cosines = torch.cos(angle_rads[:, 1::2])

    # 짝수 및 홀수 인덱스의 사인과 코사인 값을 합쳐서 pos_encoding 생성
    pos_encoding = torch.cat([sines, cosines], dim=-1)
    pos_encoding = pos_encoding.unsqueeze(0)

    return pos_encoding

  def forward(self, inputs):
    return inputs + self.pos_encoding[:, :inputs.shape[1], :]


# size
# query: (batch_size, num_heads, query_seq_len, d_model/num_heads)
# key: (batch_size, num_heads, key_seq_len, d_model/num_heads)
# value: (batch_size, num_heads, value_seq_len, d_model/num_heads)
# padding_mask: (batch_size, 1, 1, key_seq_len)
def scaled_dot_product_attention(query, key, value, mask):
  # attention score
  matmul_qk = torch.matmul(query, key.transpose(-2, -1))
  depth = key.size(-1)
  # scaling
  '''
  [ scaling ]
  스케일링 식에서
  matmul_qk은 tensor(float32-디폴트), torch.sqrt(sqrt)는 scalar타입
  => 나눗셈 연산결과 logits는 모두를 수용가능한 데이터유형인 torch.float64 데이터 타입
  => 컴퓨터 성능이 좋지 못하므로 수치정밀도와 메모리효율성 사이에서 좋은 균형을 가진 float32로 변경
  '''
  # matmul_qk는 torch.sqrt(depth) 시 스칼라 타입이므로
  logits = matmul_qk / torch.sqrt(torch.tensor(depth, dtype=torch.float32))  # 명시적으로 표시

  # masking
  if mask is not None:
    logits += (mask * -1e9)  # 마스킹할 위치에 매우 작은 음수값을 넣음

  # 매우작은 음수값이 softmax를 거치면서 0이됨
  # attention weight: (batch_size, num_heads, query_seq_len, key_seq_len)
  attention_weights = torch.nn.functional.softmax(logits, dim=-1)

  # output: (batch_size, num_heads, query_seq_len, d_model/num_heads)
  output = torch.matmul(attention_weights, value)

  return output, attention_weights


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads, name='multi_head_attention'):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    # d_model / num_heads => 논문 기준 64
    self.depth = d_model // self.num_heads

    # WQ, Wk, WV
    self.query_linear = nn.Linear(d_model, d_model)
    self.key_linear = nn.Linear(d_model, d_model)
    self.value_linear = nn.Linear(d_model, d_model)

    # WO
    self.dense = nn.Linear(d_model, d_model)

  # num_heads 개수만큼 q, w, v를 split 해주기
  def split_heads(self, inputs, batch_size):
    inputs = inputs.view(batch_size, -1, self.num_heads, self.depth)
    return inputs.permute(0, 2, 1, 3)

  def forward(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
    batch_size = query.size(0)

    # WQ, WK, WV
    query = self.query_linear(query)
    key = self.key_linear(key)
    value = self.value_linear(value)

    # 헤드 나누기
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled_dot_product_attention(query, key, value, mask)
    scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
    # (batch_size, query_seq_len, num_heads, d_model/num_heads)
    scaled_attention = scaled_attention.permute(0, 2, 1, 3)

    # 헤드 연결하기
    # (batch_size, query_seq_len, d_model)
    concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

    # WO
    # (batch_size, query_seq_len, d_model)
    outputs = self.dense(concat_attention)
    return outputs


def create_padding_mask(x):
  # 두개의 텐서를 입력으로 받아 비교해 boolean tensor 반환
  mask = torch.eq(x, 0).float()
  # (batch_size, 1, 1, key의 문장 길이)
  return mask.unsqueeze(1).unsqueeze(2)


def encoder_layer(dff, d_model, num_heads, dropout, name='encoder_layer'):
  # inputs: 임베딩된 시퀀스
  # 입력 데이터의 모양: 시퀀스의 길이는 가변적
  inputs = torch.empty(-1, d_model)
  # 패딩 마스크: 어텐션 연산 시 패딩토큰을 무시하기 위함
  padding_mask = torch.zeros((1, 1, -1))

  # 멀티-헤드 어텐션
  attention = MultiHeadAttention(d_model, num_heads, name='attention')({
    'query':inputs, 'key':inputs, 'value': inputs, 'mask': padding_mask
  })

  layerNorm
  attention = nn.Dropout(p=dropout)(attention)
  attention = nn.LayerNorm(normalized_shape=attention.size(-1), eps=1e-6)(inputs+attention)

if __name__ == '__init__':
