import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=torch.arange(position, dtype=torch.float32).unsqueeze(1),
            i=torch.arange(d_model, dtype=torch.float32).unsqueeze(0),
            d_model=d_model)

        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        sines = torch.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = torch.cos(angle_rads[:, 1::2])

        pos_encoding = torch.zeros(angle_rads.shape)
        pos_encoding[:, 0::2] = sines
        pos_encoding[:, 1::2] = cosines
        pos_encoding = pos_encoding.unsqueeze(0)

        return pos_encoding

    def forward(self, inputs):
        return inputs + self.pos_encoding[:, :inputs.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, inputs, batch_size):
        inputs = inputs.view(batch_size, -1, self.num_heads, self.depth)
        return inputs.permute(0, 2, 1, 3)

    def forward(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = query.size(0)
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        outputs = self.dense(scaled_attention)
        return outputs


def create_padding_mask(x):
    mask = (x == 0)
    return mask.unsqueeze(1).unsqueeze(2)


class EncoderLayer(nn.Module):
    def __init__(self, dff, d_model, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, padding_mask):
        attention_output = self.mha({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class Encoder(nn.Module):
    def __init__(self, num_layers, dff, d_model, num_heads, dropout):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = nn.ModuleList([EncoderLayer(dff, d_model, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, inputs, padding_mask):
        outputs = inputs
        for i in range(self.num_layers):
            outputs = self.enc_layers[i](outputs, padding_mask)
        return outputs


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    depth = key.size(-1)
    logits = matmul_qk / torch.sqrt(torch.tensor(depth, dtype=torch.float32))
    if mask is not None:
        logits += (mask * -1e9)
    attention_weights = F.softmax(logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class DecoderLayer(nn.Module):
    def __init__(self, dff, d_model, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, enc_outputs, look_ahead_mask, padding_mask):
        attention1_output = self.mha1({'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
        attention1_output = self.dropout1(attention1_output)
        out1 = self.layernorm1(inputs + attention1_output)
        attention2_output = self.mha2({'query': out1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})
        attention2_output = self.dropout2(attention2_output)
        out2 = self.layernorm2(out1 + attention2_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3


class Decoder(nn.Module):
    def __init__(self, num_layers, dff, d_model, num_heads, dropout):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.dec_layers = nn.ModuleList([DecoderLayer(dff, d_model, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, inputs, enc_outputs, look_ahead_mask, padding_mask):
        outputs = inputs
        for i in range(self.num_layers):
            outputs = self.dec_layers[i](outputs, enc_outputs, look_ahead_mask, padding_mask)
        return outputs


# class Transformer(nn.Module):
#     def __init__(self, vocab_size, num_layers, dff, d_model, num_heads, dropout):
#         super(Transformer, self).__init__()
#         self.encoder = Encoder(num_layers, dff, d_model, num_heads, dropout)
#         self.decoder = Decoder(num_layers, dff, d_model, num_heads, dropout)
#         self.final_layer = nn.Linear(d_model, vocab_size)
#
#     def forward(self, inputs, dec_inputs, enc_padding_mask, look_ahead_mask, dec_padding_mask):
#         enc_outputs = self.encoder(inputs, enc_padding_mask)
#         dec_outputs = self.decoder(dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask)
#         final_outputs = self.final_layer(dec_outputs)
#         return final_outputs

def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout):
    encoder = Encoder(num_layers, dff, d_model, num_heads, dropout)
    decoder = Decoder(num_layers, dff, d_model, num_heads, dropout)
    final_layer = nn.Linear(d_model, vocab_size)

    enc_outputs = encoder(inputs, enc_padding_mask)
    dec_outputs = decoder(dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask)
    final_outputs = final_layer(dec_outputs)
    return final_outputs
