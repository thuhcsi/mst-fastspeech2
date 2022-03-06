import argparse
import os
import re
import time
from string import punctuation
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from dataset import Dataset
from torch.utils.data import DataLoader
import utils
import hparams as hp
import audio as Audio
from text import text_to_sequence
from model.fastspeech2 import FastSpeech2
from plot.utils import plot_mel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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



if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0)

    args = parser.parse_args()

    # Get averaged speaker embedding
    mel_path = os.path.join(hp.preprocessed_path, "mel")

    names = []
    reference_mels = []


    model = get_FastSpeech2(args.step).to(device)

    for filename in tqdm(os.listdir(mel_path)[:10]):
        names.append(filename)
        reference_mels.append(np.load(os.path.join(mel_path, filename)))

    gsts = utils.get_gst(reference_mels, model)
    
    print(gsts.shape)

    os.makedirs('./preprocessed_data/ECC/gst/', exist_ok=True)
    os.makedirs('./preprocessed_data/ECC/wst/', exist_ok=True)
    

    for i in tqdm(range(len(names))):
        np.save('./preprocessed_data/ECC/gst/'+names[i][8:], gsts[i], allow_pickle=False)


    wst_features = []
    wst_weights = []

    for name in names:
        basename = name[8:-4]
        wst_feature_path = os.path.join(
            hp.preprocessed_path,
            "wst_feature",
            "{}.npy".format(basename),
        )
        wst_feature = np.load(wst_feature_path)

        wst_weight_path = os.path.join(
            hp.preprocessed_path,
            "wst_weight",
            "{}.wasr.npy".format(basename),
        )
        wst_weight = np.load(wst_weight_path)
        # print(wst_feature.shape)
        # print(wst_weight.shape)
        
        time_len = min(wst_weight.shape[1], wst_feature.shape[0])
        # print(time_len)
        wst_weight = wst_weight[:,:time_len]
        wst_feature = wst_feature[:time_len,:]

        word2phone_path = os.path.join(
            hp.preprocessed_path,
            "w2p",
            "{}.npy".format(basename),
        )
        word2phone = np.load(word2phone_path)
        
        bert_tgt_map_path = os.path.join(
            hp.preprocessed_path,
            "bt_map",
            "{}.npy".format(basename),
        )
        bert_tgt_map = np.load(bert_tgt_map_path)
        
        if wst_weight.shape[0] != word2phone.shape[0]:
            # print(basename)
            # print("~~~~~~~testing~~~~~~~~~")
            # print(wst_weight.shape, word2phone.shape)
            new_wst_weight = np.zeros([word2phone.shape[0], wst_weight.shape[1]])
            # print(bert_tgt_map)
            for tmp in range(len(bert_tgt_map)):
                new_wst_weight[bert_tgt_map[tmp]] += wst_weight[tmp]
            wst_weight = new_wst_weight
            # print(wst_weight.shape, word2phone.shape)
            # print("~~~~~~~~~end~~~~~~~~~~~~~")

        wst_features.append(wst_feature)
        wst_weights.append(wst_weight)
    
    
    wsts = utils.get_wst(wst_weights, wst_features, model)
    
    for i in tqdm(range(len(names))):
        np.save('./preprocessed_data/ECC/wst/'+names[i][8:], wsts[i], allow_pickle=False)
        
