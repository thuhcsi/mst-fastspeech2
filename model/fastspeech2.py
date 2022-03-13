from collections import OrderedDict

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import hparams as hp
from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from utils import get_mask_from_lengths
from .modules import VarianceAdaptor, ReferenceEncoder, StyleAttention, WSTEncoder, WSTAttention, LengthRegulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, speaker_num=1):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder()

    
        self.x_vec_proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_1",
                        nn.Linear(hp.x_vec_size, hp.encoder_hidden),
                    ),
                    ("relu_1", nn.ReLU()),
                    (
                        "linear_2",
                        nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
                    ),
                ]
            )
        )
        # self.adain_proj = nn.Sequential(
        #     OrderedDict(
        #         [
        #             (
        #                 "linear_1",
        #                 nn.Linear(hp.adain_emb_size, hp.encoder_hidden),
        #             ),
        #             ("relu_1", nn.ReLU()),
        #             (
        #                 "linear_2",
        #                 nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
        #             ),
        #         ]
        #     )
        # )

        # Jointly trained speaker representations
        

        # GST
        self.reference_encoder = ReferenceEncoder()
        #self.reference_encoder = WSTEncoder()
        
        self.style_attention = StyleAttention()
        self.wst_encoder = WSTEncoder()
        self.wst_attention = WSTAttention()
        self.gst_proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_1",
                        nn.Linear(hp.gst_size, hp.encoder_hidden),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("linear_2", nn.Linear(hp.encoder_hidden, hp.encoder_hidden)),
                ]
            )
        )

        self.word_to_phone = LengthRegulator()
        self.wst_proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "linear_1",
                        nn.Linear(hp.wst_size, hp.encoder_hidden),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("linear_2", nn.Linear(hp.encoder_hidden, hp.encoder_hidden)),
                ]
            )
        )
        # self.speaker_embedding = nn.Embedding(speaker_num, hp.speaker_emb_size)
        # self.speaker_proj = nn.Sequential(
        #     OrderedDict(
        #         [
        #             (
        #                 "linear_1",
        #                 nn.Linear(hp.speaker_emb_size, hp.encoder_hidden),
        #             ),
        #             ("relu_1", nn.ReLU()),
        #             ("linear_2", nn.Linear(hp.encoder_hidden, hp.encoder_hidden)),
        #         ]
        #     )
        # )
        
        self.variance_adaptor = VarianceAdaptor()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)

        self.postnet = PostNet()

    def forward(
        self,
        src_seq,
        src_len,
        mel_len=None,
        d_target=None,
        p_target=None,
        e_target=None,
        mel_target=None,
        max_src_len=None,
        max_mel_len=None,
        x_vec=None,
        use_gst=False,
        gst=None,
        use_wst=False,
        wst_feature=None,
        wst_weight=None,
        word2phone=None,
        wst=None
    ):
        start_time = time.perf_counter()
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = (
            get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        )

        encoder_output = self.encoder(src_seq, src_mask)

        if x_vec is not None:
            encoder_output = encoder_output + self.x_vec_proj(x_vec).unsqueeze(
                1
            ).expand(-1, max_src_len, -1)

        if use_gst:
            if gst is None:
                style_embedding = self.reference_encoder(mel_target)
                #print(style_embedding.shape)
                gst = self.style_attention(style_embedding)
            else:
                gst = self.style_attention(gst, set_score=True)

            encoder_output = encoder_output + self.gst_proj(gst).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        if use_wst:
            if wst is None:
                wst_style_embeddings = self.wst_encoder(wst_feature)
                # B * T * 256
                word_style_tokens = torch.bmm(wst_weight, wst_style_embeddings)
                # B * N_words * 256
                word_style_tokens = self.wst_attention(word_style_tokens)
            else:
                word_style_tokens = self.wst_attention(wst, set_score=True)
            max_phones = torch.sum(word2phone, dim=1).max()
            phone_style_tokens, phone_len = x = self.word_to_phone(word_style_tokens, word2phone, max_phones)
            # B * N_phones * 256
            #print("~~~~~~~~", "shape of phone_style_tokens is {}".format(phone_style_tokens.shape))

            encoder_output = encoder_output + self.wst_proj(phone_style_tokens)

        if d_target is not None:
            (
                variance_adaptor_output,
                d_prediction,
                d_rounded,
                p_prediction,
                e_prediction,
                _,
                _,
            ) = self.variance_adaptor(
                encoder_output,
                src_mask,
                mel_mask,
                d_target,
                p_target,
                e_target,
                max_mel_len,
            )
        else:
            (
                variance_adaptor_output,
                d_prediction,
                d_rounded,
                p_prediction,
                e_prediction,
                mel_len,
                mel_mask,
            ) = self.variance_adaptor(
                encoder_output,
                src_mask,
                mel_mask,
                d_target,
                p_target,
                e_target,
                max_mel_len,
            )

        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)

        mel_output_postnet = self.postnet(mel_output) + mel_output

        return (
            mel_output,
            mel_output_postnet,
            d_prediction,
            d_rounded,
            p_prediction,
            e_prediction,
            src_mask,
            mel_mask,
            mel_len,
        )
