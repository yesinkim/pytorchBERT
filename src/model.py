import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_positional_encoding_table(max_seq_len: int, hidden_dim: int):
    """_summary_

    Args:
        max_seq_len (int): _description_
        hidden_dim (int): _description_
    """

    def get_angle(position: int, i: int) -> float:
        return position / np.power(10000, 2 * (i // 2) / hidden_dim)

    def get_angle_vector(position: int) -> list[float]:
        return [get_angle(position, hid_j) for hid_j in range(hidden_dim)]

    pe_table = torch.Tensor([get_angle_vector(pos_i) for pos_i in range(max_seq_len)])
    pe_table[:, 0::2] = torch.sin(pe_table[:, 0::2])  # 0에서 2씩(짝수만) sin함수에 적용해줌
    pe_table[:, 1::2] = torch.cos(pe_table[:, 1::2])

    return pe_table


def get_position(sequence: Tensor) -> Tensor:
    """embedding vector의 위치 index를 반환

    Args:
        enc_input (Tensor): size: (batch_size, max_seq_len)

    Returns:
        Tensor: batch_size,
    """
    position = (
        torch.arange(sequence.size(1), device=sequence.device, dtype=sequence.dtype)
        .expand(sequence.size(0), sequence.size(1))
        .contiguous()  # size: (batch_size, max_seq_len)
    )

    return position  # (batch_size, max_seq_len)


def get_padding_mask(seq: Tensor, padding_id: int) -> Tensor:
    """attention mask for pad token

    To avoid using `pad token` in operations,
    return the masking table tensor to use when using masked_fill

    Args:
        seq_q (Tensor): input query tensor
                        tensor.size-> (batch_size, max_seq_len)
        seq_k (Tensor): input key tensor
                        tensor.size-> (batch_size, max_seq_len)
        padding_id (int)

    Returns:
        Tensor: attention pad masking table
    """
    pad_attn_mask = seq.data.eq(padding_id).unsqueeze(1)  # padding id와 같은 mask에 1번째 자리 차원 증가 size(batch_size, 1, max_seq_len)

    return pad_attn_mask.expand(seq.size(0), seq.size(1), seq.size(1)).contiguous()


class BERTEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        max_seq_len: int
        ) -> None:
        super.__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.seg_emb = nn.Embedding(2, hidden_dim)
        pe_table = get_positional_encoding_table(max_seq_len, hidden_dim)
        self.pos_emb = nn.Embedding.from_pretrained(pe_table, freeze=True)
        
    def forward(self, input_tensor: Tensor, segment_tensor: Tensor) -> Tensor:
        """_summary_

        Args:
            input_tensor (Tensor): (batch_size, max_seq_len)
            segment_tensor (Tensor): (batch_size, max_seq_len)

        Returns:
            Tensor: _description_
        """
        return self.tok_emb(input_tensor) + self.seg_emb(segment_tensor) + self.pos_emb(input_tensor)       # size: (batch_size, max_seq_len, hidden_dim)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim, dropout_rate: float = 0.0) -> None:
        """Scaled Dot Product Attention

        Args:
            head_dim (int): 각 head에서 사용할 tensor의 차원
            dropout_rate (float, optional): Literally dropout rate. Defaults to 0.0.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = head_dim ** 0.5        # head_dim없애고 싶으면 

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        score = (
            torch.matmul(query, key.transpose(-1, -2)) / self.scale
        )  # size: (batch_size, max_seq_len(len_query), max_seq_len(len_key))
        score.masked_fill_(attn_mask, -1e9)
        attn_prob = self.dropout(nn.Softmax(dim=-1)(score))
        context = torch.matmul(attn_prob, value).squeeze()

        return context, attn_prob


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0) -> None:
        super().__init__()
        assert hidden_dim // n_heads == 0
        head_dim = int(hidden_dim // n_heads)
        self.W_Q = nn.Linear(hidden_dim, n_heads)
        self.W_K = nn.Linear(hidden_dim, n_heads)
        self.W_V = nn.Linear(hidden_dim, n_heads)

        self.n_heads = n_heads
        
        self.self_attn = ScaledDotProductAttention(
            head_dim=head_dim, dropout_rate=dropout)
        self.linear = nn.Linear(n_heads, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor):
        """MultiHeadAttention

        Args:
            query (Tensor): input word vector (size: batchsize, max_seq_len(q), hidden_dim)
            key (Tensor): input word vector (size: batchsize, max_seq_len(k), hidden_dim)
            value (Tensor): input word vector (size: batchsize, max_seq_len(v), hidden_dim)
            attn_mask (Tensor): attention mask (size: batchsize, max_seq_len(q=k=v), hidden_dim)

        Returns:
            _type_: _description_
        """
        batch_size = query.size(0)

        seq_q = (
            self.W_Q(query)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, n_heads, len_q, head_dim)
        seq_s = (
            self.W_K(key)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, n_heads, len_k, head_dim)
        seq_v = (
            self.W_V(value)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, n_heads, len_v, head_dim)
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.n_heads, 1, 1
        )  # (batch_size, n_heads, len_q, len_k)
        context, _ = self.self_attn(seq_q, seq_s, seq_v, attn_mask)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.head_dim)
        )  # (batch_size, len_1, n_heads * head_dim)

        output = self.linear(context)  # (batch_size, len_q, hidden_dim)
        output = self.dropout(output)

        return output
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout=0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        n_heads: int, 
        ff_dim: int, 
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-12
        ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.ffnn = PositionWiseFeedForward(
            hidden_dim=hidden_dim, ff_dim=ff_dim, dropout=dropout
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
    
    def forward(self, enc_inputs: Tensor, enc_self_attn_mask: Tensor) -> Tensor:
        mh_output = self.layer_norm1(
            self.self_attn(
                enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
            ) + enc_inputs
        )
        ffnn_output = self.ffnn(mh_output)
        ffnn_output = self.layer_norm2(ffnn_output + mh_output)
        
        return ffnn_output


class BERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int, 
        n_heads: int,
        ff_dim: int,
        n_layers: int,
        max_seq_len: int,
        padding_id: int,
        dropout_rate: float = 0.3
        ) -> None:
        super().__init__()
        self.embedding = BERTEmbedding(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout_rate
                )
                for _ in range(n_layers)
            ]
        )
        
    def forward(self, enc_inputs: Tensor, segment_tensor: Tensor) -> Tensor:
        padding_mask = get_padding_mask(
            enc_inputs, self.padding_id
        )
        combined_emb = self.embedding(enc_inputs, segment_tensor)
        enc_output = combined_emb
        for layer in self.layers:
            enc_output = layer(enc_output, padding_mask)
        
        return enc_output