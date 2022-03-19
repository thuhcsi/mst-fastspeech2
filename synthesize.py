import argparse
import os
import re
import time
from string import punctuation
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

import utils
import hparams as hp
import audio as Audio
from text import text_to_sequence
from model.fastspeech2 import FastSpeech2
from plot.utils import plot_mel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_source(source_path):
    ids = []
    sequences = []
    with open(source_path, "r") as f:
        for line in f:
            id_, speaker, sequence = line.strip("\n").split("|")
            ids.append(id_)
            sequences.append(np.array(text_to_sequence(sequence)))
    return ids, sequences


def get_FastSpeech2(step):
    checkpoint_path = os.path.join(
        hp.checkpoint_path, "checkpoint_{}.pth.tar".format(step)
    )

    speaker_num = len(utils.get_speaker_to_id())
    model = nn.DataParallel(FastSpeech2(speaker_num))
    model.load_state_dict(torch.load(checkpoint_path)["model"])
    model.requires_grad = False
    model.eval()
    return model


def synthesize(
    model,
    vocoder,
    file_ids,
    phones,
    use_gst=False,
    gst_path=None,
    use_wst=False,
    wst_path=None,
):

    if not os.path.exists(hp.test_path):
            os.makedirs(hp.test_path)
    for i in tqdm(range(len(phones))):
        phone = phones[i]
        phone = torch.from_numpy(phone).long().to(device).unsqueeze(0)
        src_len = torch.tensor(phone.shape[1]).long().to(device).unsqueeze(0)
        id_ = file_ids[i]
        x_vec = np.load(os.path.join(hp.preprocessed_path, 'x_vec', id_+'.speaker.npy'))
        x_vec = torch.from_numpy(x_vec).float().to(device).unsqueeze(0)
        wst = None
        word2phone = None
        if use_gst:
            try:
                gst = np.load(os.path.join(gst_path, id_+'.npy'))
            except:
                continue
            gst = torch.from_numpy(gst).float().to(device).unsqueeze(0)
        if use_wst:
            try:
                wst = np.load(os.path.join(wst_path, id_+'.npy'))
            except:
                continue
            wst = torch.from_numpy(wst).float().to(device).unsqueeze(0)

            word2phone = np.load(os.path.join(hp.preprocessed_path, 'w2p', id_+'.npy'))
            word2phone = torch.from_numpy(word2phone).to(device).unsqueeze(0)

        (
            mel_output,
            mel_postnet_output,
            log_duration_output,
            _,
            f0_output,
            energy_output,
            src_mask,
            mel_mask,
            mel_len,
            p_x_vec
        ) = model(
            phone,
            src_len,
            max_src_len=phone.shape[1],
            x_vec=x_vec,
            use_gst=use_gst,
            use_wst=use_wst,
            gst=gst,
            wst=wst,
            word2phone=word2phone
        )
        utils.vocoder_infer(
            mel_postnet_output.detach().transpose(1, 2),
            vocoder,
            [os.path.join(hp.test_path, "{}.wav".format(id_))]
        )



if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=500000)
    parser.add_argument("--source", type=str, default=hp.preprocessed_path+'/eval.txt')
    parser.add_argument("--gst", action="store_true")
    parser.add_argument("--wst", action="store_true")
    parser.add_argument("--gst_path", type=str, default=hp.preprocessed_path+'/p_gst')
    parser.add_argument("--wst_path", type=str, default=hp.preprocessed_path+'/p_wst')

    args = parser.parse_args()

    file_ids, phones = read_source(args.source)


    model = get_FastSpeech2(args.step).to(device)
    vocoder = utils.get_vocoder()

    with torch.no_grad():
       synthesize(
            model,
            vocoder,
            file_ids,
            phones,
            use_gst=args.gst,
            gst_path=args.gst_path,
            use_wst=args.wst,
            wst_path=args.wst_path,
        )
