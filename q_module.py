import torch as t
import torch.nn as nn
import math


class Linear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init="linear"):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init)
        )

    def forward(self, x):
        return self.linear_layer(x)


class LatentEncoder(nn.Module):

    def __init__(self, num_hidden, num_latent, input_dim, num_self_attention_l):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(num_self_attention_l)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init="relu")
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        encoder_input = t.cat([x, y], dim=-1)

        encoder_input = self.input_projection(encoder_input)

        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        hidden = encoder_input.mean(dim=1)
        hidden = t.relu(self.penultimate_layer(hidden))

        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        std = t.exp(0.5 * log_sigma)
        eps = t.randn_like(std)
        z = eps.mul(std).add_(mu)

        return mu, log_sigma, z


class DeterministicEncoder(nn.Module):

    def __init__(self, num_hidden, num_latent, input_dim, num_self_attention_l, num_cross_attention_l):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(num_self_attention_l)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(num_cross_attention_l)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(input_dim - 1, num_hidden)
        self.target_projection = Linear(input_dim - 1, num_hidden)

    def forward(self, context_x, context_y, target_x):
        encoder_input = t.cat([context_x, context_y], dim=-1)

        encoder_input = self.input_projection(encoder_input)

        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query


class Decoder(nn.Module):

    def __init__(self, num_hidden, input_dim, num_layers_dec):
        super(Decoder, self).__init__()
        self.target_projection = Linear(input_dim - 1, num_hidden)
        self.linears = nn.ModuleList(
            [Linear(num_hidden * 3, num_hidden * 3, w_init="relu") for _ in range(num_layers_dec)]
        )
        self.final_projection = Linear(num_hidden * 3, 2)

    def forward(self, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        target_x = self.target_projection(target_x)

        hidden = t.cat([t.cat([r, z], dim=-1), target_x], dim=-1)

        for linear in self.linears:
            hidden = t.relu(linear(hidden))

        hidden = self.final_projection(hidden)

        mu_pred, sigma_pred = t.split(hidden, 1, dim=-1)
        sigma_pred = 0.1 + 0.9 * t.nn.functional.softplus(sigma_pred)

        return mu_pred, sigma_pred


class MultiheadAttention(nn.Module):

    def __init__(self, num_hidden_k):
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = t.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)

        result = t.bmm(attn, value)

        return result, attn


class Attention(nn.Module):

    def __init__(self, num_hidden, h=4):
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(
            batch_size, seq_k, self.h, self.num_hidden_per_attn
        )
        query = self.query(query).view(
            batch_size, seq_q, self.h, self.num_hidden_per_attn
        )

        key = (
            key.permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, seq_k, self.num_hidden_per_attn)
        )
        value = (
            value.permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, seq_k, self.num_hidden_per_attn)
        )
        query = (
            query.permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, seq_q, self.num_hidden_per_attn)
        )

        result, attns = self.multihead(key, value, query)
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)
        result = t.cat([residual, result], dim=-1)
        result = self.final_linear(result)
        result = self.residual_dropout(result)
        result = result + residual
        result = self.layer_norm(result)

        return result, attns
