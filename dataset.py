import math
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import hparams as hp
import hparams as hp
import audio as Audio
from utils import pad_1D, pad_2D, pad_weight
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def map_func(words_tg, words_bert):
    tag1, tag2 = 0, 0 
    l_tg = len(words_tg)
    l_bert = len(words_bert)
    ret = []
    while tag1 < l_tg and tag2 < l_bert:
        if words_tg[tag1] == words_bert[tag2]:
            ret.append(tag1)
            tag1 += 1
            tag2 += 1
            
            continue
        else:
            while tag2 < l_bert and words_bert[tag2] in words_tg[tag1]:
                tag2 += 1
                ret.append(tag1)
            tag1 += 1
    return ret

class Dataset(Dataset):
    def __init__(self, filename="train.txt", sort=True):
        self.basename, self.speaker, self.text = self.process_meta(
            os.path.join(hp.preprocessed_path, filename)
        )
        self.sort = sort
        with open(os.path.join(hp.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        while True:
            try:
                basename = self.basename[idx]
                phone = np.array(text_to_sequence(self.text[idx]))
                mel_path = os.path.join(
                    hp.preprocessed_path,
                    "mel",
                    "{}-mel-{}.npy".format(hp.dataset, basename),
                )
                mel_target = np.load(mel_path)
                D_path = os.path.join(
                    hp.preprocessed_path,
                    "alignment",
                    "{}-ali-{}.npy".format(hp.dataset, basename),
                )
                D = np.load(D_path)
                f0_path = os.path.join(
                    hp.preprocessed_path,
                    "f0",
                    "{}-f0-{}.npy".format(hp.dataset, basename),
                )
                f0 = np.load(f0_path)
                energy_path = os.path.join(
                    hp.preprocessed_path,
                    "energy",
                    "{}-energy-{}.npy".format(hp.dataset, basename),
                )
                energy = np.load(energy_path)

                x_vec_path = os.path.join(hp.preprocessed_path, "x_vec", "{}.speaker.npy".format(basename))
                x_vec = np.load(x_vec_path)
                # print(x_vec.shape)

                bert_path = os.path.join(hp.preprocessed_path, "bert", "{}.bert.npy".format(basename))
                bert = np.load(bert_path)

                if hp.use_wst:
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
                    
                    #bert_tgt_map_path = os.path.join(
                    #    hp.preprocessed_path,
                    #    "bt_map",
                    #    "{}.npy".format(basename),
                    #)
                    #bert_tgt_map = np.load(bert_tgt_map_path)
                    bert_tgt_map = np.arange(wst_weight.shape[0])
                    
                    if wst_weight.shape[0] != word2phone.shape[0]:
                        print(basename)
                        # print("~~~~~~~testing~~~~~~~~~")
                        print('wrong wst, word2phone shape:', wst_weight.shape, word2phone.shape)
                        #new_wst_weight = np.zeros([word2phone.shape[0], wst_weight.shape[1]])
                        # print(bert_tgt_map)
                        #for tmp in range(len(bert_tgt_map)):
                        #    new_wst_weight[bert_tgt_map[tmp]] += wst_weight[tmp]
                        #wst_weight = new_wst_weight
                        # print(wst_weight.shape, word2phone.shape)
                        # print("~~~~~~~~~end~~~~~~~~~~~~~")
                else:
                    wst_feature = None
                    wst_weight = None
                    word2phone = None

                sample = {
                    "id": basename,
                    "text": phone,
                    "mel_target": mel_target,
                    "D": D,
                    "f0": f0,
                    "energy": energy,
                    "x_vec": x_vec,
                    "wst_feature" : wst_feature,
                    "wst_weight" : wst_weight,
                    "word2phone" : word2phone,
                    "bert" : bert,
                }
                #print("ONE SAMPLE")
                break
            except:
                idx = (idx + 1) % self.__len__()
        
        return sample

    def process_meta(self, meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            text = []
            speaker = []
            name = []
            
            for line in f.readlines():
                n, s, t = line.strip("\n").split("|")
                # if "TSV_T2" not in s:
                name.append(n)
                speaker.append(s)
                text.append(t)
                #raw_text.append(rt)
            return name, speaker, text

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        x_vec = np.array([batch[ind]["x_vec"] for ind in cut_list])
        bert = np.array([batch[ind]["bert"] for ind in cut_list])
        if hp.use_wst:
            word2phones = [batch[ind]["word2phone"] for ind in cut_list]
            wst_features = [batch[ind]["wst_feature"] for ind in cut_list]
            wst_weights = [batch[ind]["wst_weight"] for ind in cut_list]
            
            wst_features = pad_2D(wst_features)
            word2phones = pad_1D(word2phones)
            wst_weights = pad_weight(wst_weights)
        else:
            word2phones = None
            wst_features = None
            wst_weights = None

        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        texts = pad_1D(texts)
        mel_targets = pad_2D(mel_targets)
        bert = pad_2D(bert)
        Ds = pad_1D(Ds)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + hp.log_offset)
        out = {
            "id": ids,
            "text": texts,
            "mel_target": mel_targets,
            "D": Ds,
            "log_D": log_Ds,
            "f0": f0s,
            "energy": energies,
            "src_len": length_text,
            "mel_len": length_mel,
            "x_vec": x_vec,
            "wst_feature" : wst_features,
            "wst_weight" : wst_weights,
            "word2phone" : word2phones,
            "bert" : bert,
        }

        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(
                    index_arr[i * real_batchsize : (i + 1) * real_batchsize]
                )
            else:
                cut_list.append(np.arange(i * real_batchsize, (i + 1) * real_batchsize))

        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        return output


if __name__ == "__main__":
    # Test
    dataset = Dataset("train.txt")
    training_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=0,
    )

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = (
                torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
            )
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1

    print(cnt, len(dataset))
