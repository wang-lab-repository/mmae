import torch
from torch import nn
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# word_size = 50
# w_len = 50
# v_len = 70
# a_len = 80
#
# hidden_size = 768
# max_position = 150
# modal_size = 3
# layer_norm_eps = 1e-05
# hidden_dropout_prob = 0.1  # 0.125
# word_embedding_dim = 768
# visual_dim = 20  # 20 35
# audio_dim = 5  # 5  74
# attention_dropout_prob = 0.1  # 0.125
# num_head = 8
# output_attention = 0
# num_layer = 3
# output_hidden_state = 0
# intermediate_size = 768 * 2
# num_head_modal = 8
# num_layer_modal = 0
#
# config = {'word_size': word_size, 'hidden_size': hidden_size, 'max_position': max_position,
#           'word_embedding_dim': word_embedding_dim, 'audio_dim': audio_dim, 'visual_dim': visual_dim,
#           'modal_size': modal_size, 'layer_norm_eps': layer_norm_eps, 'hidden_dropout_prob': hidden_dropout_prob,
#           'attention_dropout_prob': attention_dropout_prob, 'num_head': num_head, 'output_attention': output_attention,
#           'num_layer': num_layer, 'output_hidden_state': output_hidden_state, 'intermediate_size': intermediate_size,
#           'num_head_modal': num_head_modal, 'num_layer_modal': num_layer_modal, 'w_len': w_len, 'v_len': v_len,
#           'a_len': a_len}


class Multimodal_SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_SelfAttention, self).__init__()
        while hidden_size % 8 != 0:
            hidden_size = hidden_size + 1
        # raise ValueError(
        #     "The hidden size (%d) is not a multiple of the number of attention "
        #     "heads (%d)" % (hidden_size, 8))

        self.modal_size = 2
        self.t_len = int(hidden_size / 2)
        self.g_len = int(hidden_size / 2)
        # self.w_len = config['w_len']

        self.output_attention = 0  # 是否要输出注意力的值
        self.num_head = 8  # 多头注意力机制的数量
        self.attention_head_size = int(hidden_size / 8)  # 单头的hidden_size
        self.all_head_size = self.num_head * self.attention_head_size  # 多个头concate以后的hidden_size

        self.q = nn.Linear(hidden_size, self.all_head_size)
        self.k = nn.Linear(hidden_size, self.all_head_size)
        self.v = nn.Linear(hidden_size, self.all_head_size)

        self.dropout_1 = nn.Dropout(0.1)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=0.001)
        self.dropout_2 = nn.Dropout(0.1)

        self.w_t = nn.Parameter(torch.tensor(0.5))
        self.w_g = nn.Parameter(torch.tensor(0.5))
        # self.w_c = nn.Parameter(torch.tensor(0.2))

    def transpose1(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        x = x.view(*new_shape)  # [batch_size , sentence_len , num_head_size , attention_head_size]
        return x.permute(0, 2, 1, 3)  # [batch_size , num_head_size , sentence_len ,attention_head_size]

    def transpose2(self, x):
        new_shape = x.size()[:-2] + (x.size()[-1] * self.modal_size, int(x.size()[-1] / self.modal_size))
        x = x.view(*new_shape)
        return x

    def transpose3(self, x):
        new_shape = x.size()[:-2] + (x.size()[-1] * self.modal_size, x.size()[-1] * self.modal_size)
        x = x.view(*new_shape)
        return x

    def forward(self, hidden_state, attention_mask):
        num_head_q = self.q(hidden_state)
        num_head_k = self.k(hidden_state)
        num_head_v = self.v(hidden_state)

        q_layer = self.transpose1(num_head_q)
        k_layer = self.transpose1(num_head_k)
        v_layer = self.transpose1(num_head_v)

        attention_score = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.attention_head_size)

        # attention_score = self.transpose2(attention_score)

        # if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # my_mask = my_mask.T @ my_mask
        # my_mask = (my_mask == False)
        # attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(-1)
        # print(attention_mask.shape) batch,1,1
        attention_mask = attention_mask.permute(0, 2, 1) @ attention_mask
        # print(attention_mask.shape) batch,1,1
        my_mask = (attention_mask == False)
        my_mask = my_mask.unsqueeze(1)
        # print(my_mask.shape) batch,1,1,1

        """
        typ = attention_mask.size()[:2] + (attention_mask.size()[-1] * self.modal_size \
                , int(attention_mask.size()[-1]/self.modal_size) )
        attention_mask = attention_mask.view(*typ)        
        """

        attention_score = attention_score.masked_fill(my_mask, -1e9)
        # print(attention_score.shape)
        # [batch_size , num_head_size , sentence_len ,sentence_len]
        # print(1)
        attention_prob_t = nn.Softmax(-1)(attention_score[:, :, :, :self.t_len])
        # print(1)
        attention_prob_g = nn.Softmax(-1)(attention_score[:, :, :, self.t_len:self.t_len + self.g_len])
        # print(1)
        # attention_prob_a = nn.Softmax(-1)(
        #     attention_score[:, :, :, self.w_len + self.v_len:self.w_len + self.v_len + self.a_len])
        attention_prob = torch.cat((attention_prob_t, attention_prob_g), dim=-1)
        # print(attention_prob.shape) [32, 8, 1, 1]
        # attention_prob = self.transpose3(attention_prob)

        # m_a = torch.full((1,1,150,50),0.62).to(DEVICE)
        # m_b = torch.full((1,1,150,50),0.2).to(DEVICE)
        # m_c = torch.full((1,1,150,50),0.2).to(DEVICE)
        m_t = self.w_t * torch.ones(hidden_state.shape[0], 8, 1, 1).to(DEVICE)
        m_g = self.w_g * torch.ones(hidden_state.shape[0], 8, 1, 1).to(DEVICE)
        # m_t = self.w_t * torch.ones(1, 1, 180, hidden_size).to(DEVICE)
        # m_g = self.w_g * torch.ones(1, 1, 180, hidden_size).to(DEVICE)
        # m_c = self.w_c * torch.ones(1, 1, 180, 70).to(DEVICE)
        mask2 = m_t + m_g
        mask2 = mask2.to(DEVICE)
        # mask2 = torch.cat((m_t, m_g), dim=3).to(DEVICE)
        # print(mask2.shape) [32, 8, 1, 1]
        attention_prob = attention_prob.mul(mask2)
        # print(attention_prob.shape) [32, 8, 1, 1]
        attention_prob = self.dropout_1(attention_prob)
        # print(attention_prob.shape) [32, 8, 1, 1]
        # print(v_layer.shape) 32, 8, 1, 2
        context_layer = torch.matmul(attention_prob, v_layer)
        # print(context_layer.shape) 32, 8, 1, 2
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # print(context_layer.shape) 32, 1, 8, 2
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(*new_context_layer_shape)
        # print(context_layer.shape) 32, 1, 16
        context_layer = self.dense(context_layer)
        # print(context_layer.shape) 32, 1, 16
        context_layer = self.dropout_2(context_layer)
        hidden_state = self.LayerNorm(context_layer + hidden_state)
        # print(hidden_state.shape)
        outputs = (hidden_state, attention_prob) if self.output_attention else (hidden_state,)
        # print(tuple_shape(((1, 2), (3, 4), (5, 6))))
        # print(tuple_shape(outputs))
        # print(outputs)
        return outputs


def tuple_shape(t):
    if not isinstance(t, tuple):
        return ()
    return (len(t),) + tuple_shape(t[0])


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Feed_Forward(nn.Module):
    def __init__(self, hidden_size):
        super(Feed_Forward, self).__init__()
        while hidden_size % 8 != 0:
            hidden_size = hidden_size + 1
        self.dense_1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()

        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.Layer_Norm = nn.LayerNorm(hidden_size, eps=0.001)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_state):
        feed_forward = self.dense_1(hidden_state)
        feed_forward = self.gelu(feed_forward)
        feed_forward = self.dense_2(feed_forward)
        feed_forward = self.dropout(feed_forward)
        feed_forward = self.Layer_Norm(feed_forward + hidden_state)

        return feed_forward


class Multimodal_layer(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_layer, self).__init__()
        while hidden_size % 8 != 0:
            hidden_size = hidden_size + 1
        self.attention = Multimodal_SelfAttention(hidden_size)
        self.feed_forward = Feed_Forward(hidden_size)

    def forward(self, hidden_state, attention_mask):
        attention_output = self.attention(hidden_state, attention_mask)
        output = attention_output[0]
        output = self.feed_forward(output)
        output = (output,) + attention_output[1:]

        return output


class Mul_Encoder(nn.Module):
    def __init__(self, hidden_size=128):
        super(Mul_Encoder, self).__init__()
        while hidden_size % 8 != 0:
            hidden_size = hidden_size + 1
        self.output_attention = 0
        self.output_hidden_state = 0
        self.Layer = nn.ModuleList([Multimodal_layer(hidden_size) for _ in range(3)])

    def forward(self, hidden_state, attention_mask):
        all_hidden_state = ()
        all_attention = ()
        for i, layer_module in enumerate(self.Layer):
            if self.output_hidden_state:
                all_hidden_state = all_hidden_state + (hidden_state,)

            layer_outputs = layer_module(hidden_state, attention_mask)
            hidden_state = layer_outputs[0]

            if self.output_attention:
                all_attention = all_attention + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_state:
            all_hidden_state = all_hidden_state + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_hidden_state:
            outputs = outputs + (all_hidden_state,)
        if self.output_attention:
            outputs = outputs + (all_attention,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
