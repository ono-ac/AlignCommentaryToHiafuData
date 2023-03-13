'''
麻雀の表記を操作するためのクラス

    - num <int> :           牌の整数index
                                 11~19が萬子, 21~29が筒子, 31~39が索子, 41~47が東南西北白發中, 51が赤5萬, 52が赤5筒, 53が赤5索
                                 
    - token <str> :         牌の数字+アルファベット表記 (例 : 5p)
                                 萬子がm, 筒子がp, 索子がs, 字牌がj, 赤は先頭にrが入る(例 : r5p)
                                 
    - kana <str> :          牌のカナ呼称
                                 複数あるが最も頻度の高い呼称に統一
                                 
'''

PATH = "/home/akira/workSpace_2021/text_haifu_connect/"

# num -> token
def num2token(hainum):
    if hainum < 11:
        return ""
    haisyu = ''
    hainame = ''
    naki = ''
    if hainum // 10 == 1:
        haisyu = 'm'
    elif hainum //10 == 2:
        haisyu = 'p'
    elif hainum //10 == 3:
        haisyu = 's'
    elif hainum //10 == 4:
        haisyu = 'j'

    if hainum == 51:
        hainame = 'r5m'
    elif hainum == 52:
        hainame = 'r5p'
    elif hainum == 53:
        hainame = 'r5s'
    else:
        hainame = naki + str(int(hainum % 10)) + haisyu
    
    return hainame


def num2kana(num):
        haiNameListFileName = PATH + 'haiNameList.csv'
        dic = {}
        with open(haiNameListFileName, 'r', encoding='utf-8') as f:
            for row in f:
                tmp = row.rstrip().split(',')
                dic[int(tmp[0])] = tmp[1]
        if num not in dic:
            print(num)
        assert num in dic
        ''' This hainum does not exist. '''
        return dic[num]
                
    
    