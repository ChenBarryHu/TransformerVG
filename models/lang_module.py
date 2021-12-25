import os
import sys
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False, 
        emb_size=300, hidden_size=256, lang_num_max=32):
        super().__init__() 

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.lang_num_max = lang_num_max

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
        lang_last_list = []
        lang_scores_list = []
        for i in range(self.lang_num_max):
            word_embs = data_dict["lang_feat_list"][:, i, :, :]
            lang_feat = pack_padded_sequence(word_embs,
                                             data_dict["lang_len_list"][:, i].cpu(),
                                             batch_first=True, enforce_sorted=False)

            # encode description
            _, lang_last = self.gru(lang_feat)
            lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size, hidden_size * num_dir
            lang_last_list.append(lang_last)
            if self.use_lang_classifier:
                lang_scores = self.lang_cls(lang_last)
                lang_scores_list.append(lang_scores)
        lang_last_cat = torch.empty((lang_last.shape[0], self.lang_num_max, lang_last.shape[1]))
        if self.use_lang_classifier:
            lang_scores_cat = torch.empty((lang_scores.shape[0], self.lang_num_max, lang_scores.shape[1]))
        for i in range(self.lang_num_max - 1):
            if i == 0:
                lang_last_cat = torch.cat((lang_last_list[i].unsqueeze(1), lang_last_list[i+1].unsqueeze(1)), dim=1)
                if self.use_lang_classifier:
                    lang_scores_cat = torch.cat((lang_scores_list[i].unsqueeze(1), lang_scores_list[i+1].unsqueeze(1)), dim=1)
            else:
                lang_last_cat = torch.cat((lang_last_cat, lang_last_list[i+1].unsqueeze(1)), dim=1)
                if self.use_lang_classifier:
                    lang_scores_cat = torch.cat((lang_scores_cat, lang_scores_list[i+1].unsqueeze(1)), dim=1)

        # store the encoded language features
        data_dict["lang_emb"] = lang_last_cat # B, lang_num_max, hidden_size

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = lang_scores_cat


        return data_dict

