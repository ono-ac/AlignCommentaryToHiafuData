import os

import torch
import numpy as np
import pandas as pd
from classopt import classopt
import json
import datetime

from transformers import BertForSequenceClassification, BertTokenizer

from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

import MahjongTool

@classopt(default_long=True)
class Args:
    model_path:str = "train_2023_3_14_15_24_31/2023_3_14_15_24_31_4ep.pth"
    device:str = "cuda:0"
    haifu_dir:str = "haifuData"
    commentary_dir:str = "LinkedText"
    output_dir:str = "test_{}"
    use_script_fix_file:bool = False
    debug:bool = False
        
def main(args: Args):
    model_path:str = args.model_path
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    haifu_dir = args.haifu_dir
    output_dir = args.output_dir
    commentary_dir = args.commentary_dir
    use_script_fix_file = args.use_script_fix_file
    dt_now = datetime.datetime.now()
    DAY = "{}_{}_{}_{}_{}_{}".format(dt_now.year, dt_now.month, dt_now.day, dt_now.hour, dt_now.minute, dt_now.second)
    debug = args.debug
    
    
    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', return_dict=True)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    
    filenums = ["example_test"]
    
    def fileOpener(filenums, use_fix_file:bool = False):
        haifuDatas = {}
        for filenum in filenums:
            haifuData = []
            with open(haifu_dir + "/" + '{}.json'.format(filenum), 'r', encoding='utf-8') as cf:
                for raw in cf:
                    j_tmp = json.loads(raw.rstrip())
                    haifuData.append(j_tmp)
            haifuDatas[filenum] = haifuData

        kaisetu_texts = {}
        for filenum in filenums:
            g_filename = commentary_dir  + "/" + filenum + "_linkedtext.csv"
            sents = []
            with open(g_filename, 'r', encoding='utf-8') as fg:
                for _raw in fg:
                    tmp = _raw.rstrip().split(',')
                    sent_text = tmp[2]
                    fix_sent_text = sent_text
                    if (use_script_fix_file):
                        fix_sent_text = _FixMahjongWord(sent_text, fix_mahjong_word_dic)
                        fix_sent_text = _FixVR(fix_sent_text, fix_vr_error_dic)
                    
                    sents.append(fix_sent_text)
        kaisetu_texts[filenum] = sents
        print(kaisetu_texts)

        return {'haifuDatas': haifuDatas, 'kaisetu_texts': kaisetu_texts}
    
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

    def softmax(x):
        if (x.ndim == 1):
            x = x[None,:]   
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
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
    
    def BertBaseScore(sent, haifu):
        if len(sent) > 500:
            sent = sent[:500]
        haifu_text = MakeHaifuInput(haifu)
        encoding = tokenizer(sent, haifu_text, return_tensors='pt', padding='max_length', max_length = 500)

        input_ids = torch.tensor(encoding['input_ids'])
        token_type_ids = torch.tensor(encoding['token_type_ids'])
        attention_mask = torch.tensor(encoding['attention_mask'])

        t_input_ids = input_ids.to(device)
        t_token_type_ids = token_type_ids.to(device)
        t_attention_mask = attention_mask.to(device)

        preds = model(t_input_ids, 
                     token_type_ids=t_token_type_ids, 
                     attention_mask=t_attention_mask)

        ta = [preds.logits[0][0].item(), preds.logits[0][1].item()]
        y = softmax(np.array(ta))
        return y[0][1]
    
    if debug:
        text = '瑞原これでテンパイ。'
        haifu = {"index": 1, "player": "瑞原", "action": "normal", "draw": 45, "discard": 42, "tehai_before": [14, 21, 23, 23, 24, 27, 28, 28, 31, 32, 39, 42, 44, 45], "tehai_after": [14, 21, 23, 23, 24, 27, 28, 28, 31, 32, 39, 44, 45], "special_tag": "-"}
        print(BertBaseScore(text, haifu))
        
    def Link(haifuData, kaisetuText, haifuInfo):
        rute = Link_BertBase(haifuData, kaisetuText, haifuInfo)
        return rute
    
    def Link_BertBase(haifuData, kaisetuText, haifuInfo):
        dp = [[0 for _ in range(len(haifuData))] for _ in range(len(kaisetuText)+1)]    #dp[text][haifu]
        rute = [[[] for _ in range(len(haifuData))] for _ in range(len(kaisetuText)+1)]
        haifu_id = 0
        text_id = 0

        for text_id in range(1, len(kaisetuText)+1):  
            for haifu_id in range(len(haifuData)):
                if haifu_id == 0:
                    dp[text_id][haifu_id] = dp[text_id-1][haifu_id] + BertBaseScore(kaisetuText[text_id-1], haifuData[haifu_id])
                    rute[text_id][haifu_id] = rute[text_id-1][haifu_id] + [haifu_id]
                else:
                    #up shift
                    if dp[text_id][haifu_id-1] >= dp[text_id-1][haifu_id] + BertBaseScore(kaisetuText[text_id-1], haifuData[haifu_id]):
                        dp[text_id][haifu_id] = dp[text_id][haifu_id-1]
                        rute[text_id][haifu_id] = rute[text_id][haifu_id-1]
                    #right shift
                    else:
                        dp[text_id][haifu_id] = dp[text_id-1][haifu_id] + BertBaseScore(kaisetuText[text_id-1], haifuData[haifu_id])
                        rute[text_id][haifu_id] =  rute[text_id-1][haifu_id] + [haifu_id]
            
        return rute[len(kaisetuText)][len(haifuData)-1]
   
    def LinkText2Haifu(haifuDatas, kaisetuTexts, filenums):
    
        result_table = []
        for filenum in filenums:

            haifuData = haifuDatas[filenum]
            kaisetuText = kaisetuTexts[filenum]
            haifuInfo = {}
            print('【{}】'.format(filenum))

            linkList = Link(haifuData, kaisetuText, haifuInfo)
            SaveResult(filenum, kaisetuText, linkList)
            #score_table = ScoreTabler(haifuData, kaisetuText, filenum)
            
            dic = {}
            #dic = Acc(filenum)
            result_table.append(dic)
            cnt = 0

        #for t in result_table:
            #cnt += t['correct']
        #print('-------------------------------->  total correct : {}'.format(cnt))

        return result_table
    
    def SaveResult(filenum, kaisetuText, linkList):
        writething = ''
        if not os.path.exists(output_dir.format(DAY)):
            os.mkdir(output_dir.format(DAY))
        for i in range(len(kaisetuText)):
            writething += str(linkList[i]) + ',' + str(i) + ',' + kaisetuText[i] + '\n'
        with open(output_dir.format(DAY) + "/" + 'link_output_{}.csv'.format(DAY), 'w', encoding='utf-8') as wf:
            wf.write(writething)
    
    # align
    fp = fileOpener(filenums)
    haifuDatas = fp['haifuDatas']
    kaisetu_texts = fp['kaisetu_texts']
    relults = []
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(model_path))
    result = LinkText2Haifu(haifuDatas, kaisetu_texts, filenums)
    relults.append(result)
    print(result)
    
    
if __name__ == "__main__":
    args = Args.from_args()
    main(args)