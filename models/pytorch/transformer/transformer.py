import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
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
  matmul_pk = torch.matmul(query, key.t())


