import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, SequentialSampler, RandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import os
from classopt import classopt
import json
import re
import random
import collections
import datetime
import warnings

from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

import MahjongTool

@classopt(default_long=True)
class Args:
    detail:bool = False
    device:str = "cuda:0"
    commentary_dir:str = "LinkedText"
    haifu_dir:str = "haifuData"
    save_output_dir:str = "train_{}"
    script_fix_file:str = ""
    neg_per_pos:int = 15
    batch_size:int = 8
    max_epoch:int = 5
    lr:float = 1e-5
    output_id:str = ""
    save_all_epoch_model:bool = True
    debug:bool = False

def main(args: Args):
    warnings.simplefilter('ignore', FutureWarning)
    
    detail = args.detail
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    commentary_dir = args.commentary_dir
    haifu_dir = args.haifu_dir
    use_script_fix_file = False
    if args.script_fix_file != "":
        use_script_fix_file = True
    save_all_epoch_model = args.save_all_epoch_model
    save_output_dir = args.save_output_dir
    debug = args.debug
    
    INC_PER = args.neg_per_pos
    BATCH_SIZE = args.batch_size
    max_epoch = args.max_epoch
    lr = args.lr
    dt_now = datetime.datetime.now()
    DAY = "{}_{}_{}_{}_{}_{}".format(dt_now.year, dt_now.month, dt_now.day, dt_now.hour, dt_now.minute, dt_now.second)
    if args.output_id != "":
            DAY = args.output_id
    
    
    filenums4train = ["example_train"]
    filenums4test = ["example_dev"]
    filenums = filenums4train + filenums4test 
    print(filenums)
    
    gold_link_filenames = [commentary_dir + '/' + filenum + "_linkedtext.csv" for filenum in filenums]
    haifu_filenames = [haifu_dir  + '/' + filenum + ".json" for filenum in filenums]
    sents = {}
    haifus = {}
    
    if use_script_fix_file:
        with open("forNames2Name.json", 'r', encoding='utf-8') as jf:
            fix_mahjong_word_dic = json.load(jf)

        with open("forFixVrError.json", 'r', encoding='utf-8') as ffve:
            fix_vr_error_dic = json.load(ffve)
    
    def _FixMahjongWord(text, fixMahjongWordDic):
        for main_name, candicate_name_list in fixMahjongWordDic.items():
            for cand_name in candicate_name_list:
                text = re.sub(cand_name, main_name, text)
        return text

    def FixMahjongWord(texts, fixMahjongWordDic):
        new_texts = []
        for t in texts:
            new_texts.append(_FixMahjongWord(t, fixMahjongWordDic))
        return new_texts

    def _FixVR(text, fixVRWordDic):
        for main_name, candicate_name_list in fixVRWordDic.items():
            for cand_name in candicate_name_list:
                text = re.sub(cand_name, main_name, text)
        return text

    def FixVR(texts, fixVRWordDic):
        new_texts = []
        for t in texts:
            new_texts.append(_FixMahjongWord(t, fixVRWordDic))
        return new_texts
    
    for filenum in filenums:
        g_filename = commentary_dir + '/' + filenum + "_linkedtext.csv"
        sentdic_list = []
        
        with open(g_filename, 'r', encoding='utf-8') as fg:
            for _raw in fg:
                tmp = _raw.rstrip().split(',')
                sent_text = tmp[2]
                fix_sent_text = sent_text
                
                if (use_script_fix_file):
                    fix_sent_text = _FixMahjongWord(sent_text, fix_mahjong_word_dic)
                    fix_sent_text = _FixVR(fix_sent_text, fix_vr_error_dic)
                    
                sent_dic = {'sent' : fix_sent_text, 'haifu_num' : tmp[0]}
                sentdic_list.append(sent_dic)
        sents[filenum] = sentdic_list

   
    haifu_indexs = {}
    for filenum in filenums:
        haifu_datas = []
        haifu_index = []
        h_filename = haifu_dir  + "/" + filenum + ".json"
        with open(h_filename, 'r', encoding='utf-8') as hg:
            for _raw in hg:
                js = _raw.rstrip()
                haifu_data = json.loads(js)
                haifu_datas.append(haifu_data)
                haifu_index.append(haifu_data['new_id'])
        haifus[filenum] = haifu_datas
        haifu_indexs[filenum] = haifu_index
    
    t_sent_haifu_label = []
    for filenum in filenums4train:
        for sent in sents[filenum]:
            if sent['haifu_num'] != '-':
                h_num = int(sent['haifu_num'])
                bit = {'sent' : sent['sent'], 'haifu' : haifus[filenum][h_num], 'label' : 1}
                t_sent_haifu_label.append(bit)
    
    f_sent_haifu_label = []
    for filenum in filenums4train:
        for sent in sents[filenum]:
            if sent['haifu_num'] != '-':
                ex_haifu_index_list = haifu_indexs[filenum]
                index = ex_haifu_index_list.index(int(sent['haifu_num']))
                rd_ids = random.sample(list(range(1, len(ex_haifu_index_list))), INC_PER)
                for rd_id in rd_ids:
                    rd = (index + rd_id) % len(ex_haifu_index_list)
                    bit = {'sent' : sent['sent'], 'haifu' : haifus[filenum][ex_haifu_index_list[rd]], 'label' : 0}
                    f_sent_haifu_label.append(bit)    
    
    sent_haifu_label = f_sent_haifu_label + t_sent_haifu_label
    
    def TehaiShinkou(tehai):
        tmp = ['', '', '', '']
        shanten = Shanten()
        for t in tehai:
            if t // 10 == 1:
                tmp[0] += str(int(t % 10))
            elif t // 10 == 2:
                tmp[1] += str(int(t % 10))
            elif t // 10 == 3:
                tmp[2] += str(int(t % 10))
            elif t // 10 == 4:
                tmp[3] += str(int(t % 10))
            elif t // 10 == 5:
                tmp[int(t % 10) - 1] += str(5) 
        tiles = TilesConverter.string_to_34_array(man=tmp[0], pin=tmp[1], sou=tmp[2], honors=tmp[3])
        result = shanten.calculate_shanten(tiles)
        return result

    def shanten2word(shanten):
        s2w = {
                                    -1:"ツモ", 
                                    0: "テンパイ",
                                    1: "イーシャンテン", 
                                    2: "リャンシャンテン", 
                                    3: "サンシャンテン", 
                                    4: "スーシャンテン", 
                                    5:"ウーシャンテン", 
                                    6: "ローシャンテン"
                    }
        return s2w[shanten]
    
    def MakeHaifuInput(haifu):
        tag2word = {'reach': 'リーチ', 'chi': 'チー', 'pon': 'ポン', 'normal': ''}
        if haifu['action'] == 'start':
            return '[action] 開始'

        elif haifu['action'] == 'finish':
            rtn = '[action] 終了 '
            if haifu['special_tag'] == 'ryukyoku':
                rtn += '流局'
            elif haifu['special_tag'] == 'tumo':
                rtn += 'ツモ' + ' [player] ' + haifu['player'] + ' [tumo] ' + str(MahjongTool.num2kana(haifu['draw']))
            else:
                rtn += 'ロン' + ' [player] ' + haifu['player']
            return rtn

        else:
            rtn = '[player] ' + haifu['player']

            if haifu['action'] != 'normal':
                rtn +=  ' [action] ' + tag2word[haifu['action']]

            rtn += ' [tumo] ' + str(MahjongTool.num2kana(haifu['draw'])) + ' [dahai] ' + str(MahjongTool.num2kana(haifu['discard']))
            rtn += ' [shanten] ' + shanten2word(TehaiShinkou(haifu['tehai_before']))

        return rtn
    
    train_haifu_strings = []

    for shl in sent_haifu_label:
        thst = MakeHaifuInput(shl['haifu'])
        train_haifu_strings.append(thst)
    
    train_sents = []
    for shl in sent_haifu_label:
        ts = shl['sent']
        train_sents.append(ts)

    train_labels = [shl['label'] for shl in sent_haifu_label]

    c = collections.Counter(train_labels)
    
    # test set
    t_sent_haifu_label4test = []
    for filenum in filenums4test:
        for sent in sents[filenum]:
            if sent['haifu_num'] != '-':
                h_num = int(sent['haifu_num'])
                bit = {'sent' : sent['sent'], 'haifu' : haifus[filenum][h_num], 'label' : 1}
                t_sent_haifu_label4test.append(bit)

    f_sent_haifu_label4test = []

    for filenum in filenums4test:
        for sent in sents[filenum]:
            if sent['haifu_num'] != '-':
                ex_haifu_index_list = haifu_indexs[filenum]
                index = ex_haifu_index_list.index(int(sent['haifu_num']))
                rd_ids = random.sample(list(range(1, len(ex_haifu_index_list))), 1)
                for rd_id in rd_ids:
                    rd = (index + rd_id) % len(ex_haifu_index_list)
                    bit = {'sent' : sent['sent'], 'haifu' : haifus[filenum][ex_haifu_index_list[rd]], 'label' : 0}
                    f_sent_haifu_label4test.append(bit)   

    sent_haifu_label4test = f_sent_haifu_label4test + t_sent_haifu_label4test

    haifu_strings4test = []

    for shl in sent_haifu_label4test:
        thst = MakeHaifuInput(shl['haifu'])
        haifu_strings4test.append(thst)

    test_sents = []
    for shl in sent_haifu_label4test:
        ts = shl['sent']
        test_sents.append(ts)

    test_labels = [shl['label'] for shl in sent_haifu_label4test]

    c = collections.Counter(test_labels)
    
    # load model
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', return_dict=True)
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    
    encoding = tokenizer(train_haifu_strings, train_sents, return_tensors='pt', padding=True, max_length=300)
    input_ids = encoding['input_ids']
    token_type_ids = encoding['token_type_ids']
    attention_mask = encoding['attention_mask']
    
    encoding4test = tokenizer(haifu_strings4test, test_sents, return_tensors='pt', padding=True, max_length=300)
    input_ids4test = encoding4test['input_ids']
    token_type_ids4test = encoding4test['token_type_ids']
    attention_mask4test = encoding4test['attention_mask']
    
    if debug:
        print("---------------------【debug】---------------------")
        print(" - input ids - ")
        print(input_ids[1])
        print(" - token type ids - ")
        print(token_type_ids[1])
        print(" - attention mask - ")
        print(attention_mask[1])
        print(" - token decode - ")
        print(tokenizer.decode(input_ids[1]))
        
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    train_labels = torch.tensor(train_labels)
    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, train_labels)
    
    train_size = int(1.0 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print('train：{}'.format(train_size))
    
    batch_size = BATCH_SIZE

    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size
            )
    
    input_ids4test = torch.tensor(input_ids4test)
    attention_mask4test = torch.tensor(attention_mask4test)
    token_type_ids4test = torch.tensor(token_type_ids4test)
    test_labels = torch.tensor(test_labels)

    test_dataset = TensorDataset(input_ids4test, token_type_ids4test, attention_mask4test, test_labels)

    print('val：{}'.format(len(test_dataset)))

    batch_size = BATCH_SIZE

    test_dataloader = DataLoader(
                test_dataset,  
                sampler = SequentialSampler(test_dataset), 
                batch_size = batch_size
            )
    
    def train_model(model):
        model.train()
        train_loss = 0

        for batch in train_dataloader:
            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)
            optimizer.zero_grad()
            outputs = model(b_input_ids, 
                                 token_type_ids=b_token_type_ids, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        return train_loss


    def validation(model):
        model.eval()
        val_loss = 0
        al, ac = 0, 0
        data = [0] * 8
        df = pd.DataFrame()
        with torch.no_grad(): 
            for batch in test_dataloader:
                b_input_ids = batch[0].to(device)
                b_token_type_ids = batch[1].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device)
                with torch.no_grad():        
                    preds = model(b_input_ids, 
                                        token_type_ids=b_token_type_ids, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)

                    val_loss += preds.loss.item()
                    pred_ids = []
                    for n in range(len(b_labels)):
                        ta = [preds.logits[n][0].item(), preds.logits[n][1].item()]
                        if ta[0] > ta[1]:
                            pred_id = 0
                        else:
                            pred_id = 1
                        pred_ids.append(pred_id)
                    for n in range(len(b_labels)):
                        data[0] += 1
                        if pred_ids[n] == b_labels[n]:
                            data[5] += 1
                            if b_labels[n] == 0:
                                data[2] += 1     # True Negative
                            else:
                                data[1] += 1     # True Positive
                        else:
                            if b_labels[n] == 0:
                                data[4] += 1     # False Neganive
                            else:
                                data[3] += 1     # False Positive

        data[6] = data[0] - data[5]
        data[7] = data[5] / data[0]

        df = pd.DataFrame([data], columns=["size", "TP", "TN", "FP", "FN", "True", "False", "acc"], index=["epoch"])

        print("-----------------------------------")
        print(" {} / {} ".format(data[5], data[0]))
        print("val accuracy :       {}".format(data[5]/data[0]))
        print("val loss :       {}".format(val_loss))
        val_accs.append(data[5]/data[0])
        return data
    
    def softmax(x):
        if (x.ndim == 1):
            x = x[None,:]  
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    
    # model initialize
    
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', return_dict=True)
    model.train() 
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5) 
    
    train_loss_ = []
    test_loss_ = []
    val_accs = []
    df = pd.DataFrame(0, columns=["size", "TP", "TN", "FP", "FN", "True", "False", "acc"], index=["epoch"])
    df[['size', 'TP', 'TN', 'FP', 'FN', 'True', 'False']] = df[['size', 'TP', 'TN', 'FP', 'FN', 'True', 'False']].astype('int')
    
    if not os.path.exists(save_output_dir.format(DAY)):
        os.mkdir(save_output_dir.format(DAY))
    
    for epoch in range(1, max_epoch + 1):
        model.train()
        train_ = train_model(model)
        print("++++++++++++++++++++ epoch {} ++++++++++++++++++++".format(epoch))
        print("training loss : \t\t {}".format(train_))

        # validation
        data = validation(model)
        df.loc['epoch {}'.format(epoch)] = data
        df[['size', 'TP', 'TN', 'FP', 'FN', 'True', 'False']] = df[['size', 'TP', 'TN', 'FP', 'FN', 'True', 'False']].astype('int')
        train_loss_.append(train_)
        
        model_path_per_epoch = save_output_dir.format(DAY) + '/' + DAY + '_{}ep.pth'.format(epoch)
        if save_all_epoch_model:
            torch.save(model.state_dict(), model_path_per_epoch)
            print("save model > {}".format(model_path_per_epoch))

        print("\n")
    
    df.to_csv(save_output_dir.format(DAY) + '/' + 'output_{}.csv'.format(DAY))


if __name__ == "__main__":
    args = Args.from_args()
    main(args)

