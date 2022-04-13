import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p

from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon("text/librispeech-lexicon.txt")

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones))

    return np.array(sequence)



def get_word2phone(text, phone_list):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon("text/librispeech-lexicon.txt")

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    tag1 = 0
    tag2 = 0
    word2phone = []
    phone_list = phone_list[1:-1].split(' ')
    max_len = len(phone_list)
    for w in words:
        ans_now = 0
        if phone_list[tag1] == 'sil':
            tag1 += 1
            ans_now += 1
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
        while tag1 < max_len and tag2 < len(phones) and phone_list[tag1] == phones[tag2]:
            tag1 += 1
            tag2 += 1
            ans_now += 1
        word2phone.append(ans_now)
    print(' '.join(phones))
    
    return np.array(word2phone)



if __name__ == "__main__":
    PATH = 'ECC_PATH'
    with open(PATH + 'trains.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split('|')
            print(line[3])
            print(line[2])
            t = get_word2phone(line[3], line[2])
            print(t)

            break
