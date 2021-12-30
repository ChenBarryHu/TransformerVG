import os
import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, dropout= 0.1, max_len = 126):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LangModuleBert(nn.Module):
    def __init__(
        self,
        num_text_classes, 
        use_lang_classifier=True,
        embed_dim=768,
        num_head=4,
        dropout=0.1,
        batch_first=True,
        lang_dim = 128
    ):
        super(LangModuleBert, self).__init__()
        self.d_model = embed_dim
        self.lang_dim = lang_dim
        self.use_lang_classifier = use_lang_classifier
        self.lang_projection = nn.Sequential(
            nn.Linear(self.d_model, self.lang_dim),
        )
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(embed_dim, num_text_classes),
            )
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
    def forward(self, data_dict):
        input_ids = data_dict["bert_input_ids"]
        token_type_ids = data_dict["bert_token_type_ids"]
        attention_mask = data_dict["bert_attention_mask"]
        output = self.bert(input_ids, token_type_ids, attention_mask)
        data_dict["attention_mask"] = (attention_mask == 0)
        data_dict['lang_emb'] = self.lang_projection(output[0])
        global_lang_feature = output[1]
        global_lang_feature = global_lang_feature.squeeze(-1)
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(global_lang_feature)
        return data_dict


    
class LangModuleTransEncoder(nn.Module):
    def __init__(
        self,
        num_text_classes, 
        use_lang_classifier=True,
        embed_dim=300,
        num_head=4,
        dropout=0.1,
        batch_first=True,
        lang_dim = 128
    ):
        super(LangModuleTransEncoder, self).__init__()
        self.d_model = embed_dim
        self.lang_dim = lang_dim
        self.pe = PositionalEmbedding(lang_dim)
        self.num_head = num_head
        self.use_lang_classifier = use_lang_classifier
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=lang_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        # self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        # self.dropout = nn.Dropout(dropout)
        # self.src_pad_idx = src_pad_idx

        # project the lang features from 300 to 128
        self.lang_projection = nn.Sequential(
            nn.Linear(self.d_model, self.lang_dim),
        )

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_dim, num_text_classes),
            )

    def lang_len_to_mask(self, lang_len, max_len=126, dtype=torch.bool):
        """ Create key padding mask for lang features' self-attention.
        lang_len: shape: (Batch_size)
        max_len: maximum number of words in a description sentence.
        dtype: data type of the output mask, default: bool. 

        output: key_padding_mask with shape: (Batch_size, max_len) dtype:torch.bool.
                For key_padding_mask, "True" value indicates "do not attend".
        """
        assert len(lang_len.shape) == 1, 'lang_len shape should be 1 dimensional.'
        max_len = max_len or lang_len.max().item()
        mask = torch.arange(max_len, device=lang_len.device,
                            dtype=lang_len.dtype).expand(len(lang_len), max_len) >= lang_len.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=lang_len.device)
        return mask
    
    def forward(self, data_dict):
        word_embedding = self.lang_projection(data_dict["lang_feat"]) # (batch_size, MAX_DES_LEN=126)
        word_embedding = word_embedding.permute(1,0,2)
        word_embedding_with_pos = self.pe(word_embedding)
        # word_embedding_with_pos = word_embedding_with_pos.permute(1,0,2)
        key_padding_mask = self.lang_len_to_mask(data_dict["lang_len"])
        data_dict["attention_mask"] = key_padding_mask
        embedding = self.transformer_encoder(src=word_embedding_with_pos, src_key_padding_mask=key_padding_mask)
        embedding = self.transformer_encoder(src=word_embedding_with_pos)
        data_dict["lang_emb"] = embedding.permute(1,0,2)
        global_lang_feature = F.max_pool2d(
            embedding.permute(1, 2, 0), kernel_size=[1, 126]
        )
        global_lang_feature = global_lang_feature.squeeze(-1)
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(global_lang_feature)

        return data_dict


class LangModuleAttention(nn.Module):
    def __init__(
        self,
        num_text_classes, 
        use_lang_classifier=True,
        embed_dim=300,
        num_head=4,
        dropout=0.1,
        batch_first=True,
        lang_dim = 128,
        use_fc=False
    ):
        super(LangModuleAttention, self).__init__()
        self.d_model = embed_dim
        self.lang_dim = lang_dim
        self.pe = PositionalEmbedding(lang_dim)
        self.num_head = num_head
        self.use_lang_classifier = use_lang_classifier
        self.self_attention = nn.MultiheadAttention(
            embed_dim=lang_dim, 
            num_heads=self.num_head,
            dropout=dropout,
            batch_first=True
        )

        # project the lang features from 300 to 128
        self.lang_projection = nn.Sequential(
                nn.Linear(self.d_model, self.lang_dim),
            )
        self.use_fc = use_fc
        if use_fc:
            self.fc = nn.Linear(lang_dim, lang_dim)

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_dim, num_text_classes),
            )

    def lang_len_to_mask(self, lang_len, max_len=126, dtype=torch.bool):
        """ Create key padding mask for lang features' self-attention.
        lang_len: shape: (Batch_size)
        max_len: maximum number of words in a description sentence.
        dtype: data type of the output mask, default: bool. 

        output: key_padding_mask with shape: (Batch_size, max_len) dtype:torch.bool.
                For key_padding_mask, "True" value indicates "do not attend".
        """
        assert len(lang_len.shape) == 1, 'lang_len shape should be 1 dimensional.'
        max_len = max_len or lang_len.max().item()
        mask = torch.arange(max_len, device=lang_len.device,
                            dtype=lang_len.dtype).expand(len(lang_len), max_len) >= lang_len.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=lang_len.device)
        return mask
    
    def forward(self, data_dict):
        word_embedding = self.lang_projection(data_dict["lang_feat"]) # (batch_size, MAX_DES_LEN=126)
        word_embedding = word_embedding.permute(1,0,2)
        word_embedding_with_pos = self.pe(word_embedding)
        word_embedding_with_pos = word_embedding_with_pos.permute(1,0,2)
        key_padding_mask = self.lang_len_to_mask(data_dict["lang_len"])
        data_dict["attention_mask"] = key_padding_mask
        embedding = self.self_attention(word_embedding_with_pos, word_embedding_with_pos, word_embedding_with_pos, key_padding_mask=key_padding_mask)
        if self.use_fc:
            data_dict["lang_emb"] = self.fc(embedding[0])
        else:
            data_dict["lang_emb"] = embedding[0]
        global_lang_feature = F.max_pool2d(
            embedding[0].permute(0, 2, 1), kernel_size=[1, 126]
        )
        global_lang_feature = global_lang_feature.squeeze(-1)
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(global_lang_feature)

        return data_dict


        

class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False, 
        emb_size=300, hidden_size=256):
        super().__init__() 

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )


    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        word_embs = data_dict["lang_feat"]
        lang_feat = pack_padded_sequence(word_embs, data_dict["lang_len"].cpu(), batch_first=True, enforce_sorted=False)
    
        # encode description
        _, lang_last = self.gru(lang_feat)
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size, hidden_size * num_dir

        # store the encoded language features
        data_dict["lang_emb"] = lang_last # B, hidden_size
        
        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict

