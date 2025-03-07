import traceback
import time
import json
import sys
import os

path_sys = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_sys)
print(path_sys)

# from char_similar.const_dict import dict_char_component, dict_char_fourangle
# from char_similar.const_dict import dict_char_frequency, dict_char_number
from char_similar.const_dict import dict_char_pinyin
# from char_similar.const_dict import dict_char_struct, dict_char_order
# from char_similar.const_dict import load_json, save_json

path_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(path_root)
path_char_pinyin = os.path.join(path_root, "data/char_pinyin.dict")


def find_sim_word_in_list(str_list, target, limit=10, code=4, rounded=2):
    """ 计算待检测词语列表与target（热词）的相似度, 通过两个词语的拼音(拼音/声母/韵母/声调)
    calculate similarity of two chars, by char pinyin
    Args:
        list: list(string)
        target: string
    Returns:
        result: list(dict{"word", "score"})
    """
    score = []
    for word in str_list:
        score.append(sim_word(word, target, code=code, rounded=rounded))

    # 将score和str_list配对
    paired_list = list(zip(score, str_list))

    # 按照score从大到小排序
    sorted_paired_list = sorted(paired_list, key=lambda x: x[0], reverse=True)

    # 获取前limit个元素
    top_limit = sorted_paired_list[:limit]
    dict_list = []
    for item in top_limit:
        data = {"word": item[1], "score": (1 - item[0]) * 10}
        dict_list.append(data)
    return dict_list


def calculate_similarity(w1, w2, length, code=4, rounded=4):
    score = 0
    for char1, char2 in zip(w1, w2):
        score += sim_pinyin(char1, char2, code=code, rounded=rounded)
    return score / length


def sim_word(word1, word2, code=4, rounded=4):
    """ 计算两个词语的相似度, 通过两个词语的拼音(拼音/声母/韵母/声调)
    calculate similarity of two chars, by char pinyin
    Args:
        word1: string, eg. "示范"
        word2: string, eg. "试戴"
    Returns:
        result: float, 0-1, eg. 0.75
    """
    len1, len2 = len(word1), len(word2)
    length = max(len1, len2)

    if len1 == len2:
        result = calculate_similarity(word1, word2, length, code, rounded)
    elif len1 > len2:
        max_score = 0
        for i in range(len1 - len2 + 1):
            sub_word1 = word1[i:i + len2]
            max_score = max(
                max_score,
                calculate_similarity(sub_word1, word2, length, code, rounded))
        result = max_score
    else:
        max_score = 0
        for i in range(len2 - len1 + 1):
            sub_word2 = word2[i:i + len1]
            max_score = max(
                max_score,
                calculate_similarity(word1, sub_word2, length, code, rounded))
        result = max_score

    return round(result, rounded)


def sim_pinyin(char1, char2, code=4, rounded=4):
    """ 计算两汉字相似度, 通过两个字拼音(拼音/声母/韵母/声调)
    calculate similarity of two chars, by char pinyin
    Args:
        char1: string, eg. "一"
        char2: string, eg. "而"
    Returns:
        result: float, 0-1, eg. 0.75
    """

    char1_pi = dict_char_pinyin.get(char1, [])
    char2_pi = dict_char_pinyin.get(char2, [])
    result = 0
    if char1_pi and char2_pi:
        same_count = sum(
            1 for cp1, cp2 in zip(char1_pi, char2_pi) if cp1 == cp2)
        result = same_count / code

    result = round(result, rounded)
    return result


def load_json(path, parse_int=None):
    """
        加载json
    Args:
        path_file[String]:, path of file of save, eg. "corpus/xuexiqiangguo.lib"
        parse_int[Boolean]: equivalent to int(num_str), eg. True or False
    Returns:
        data[Any]
    """
    with open(path, mode="r", encoding="utf-8") as fj:
        model_json = json.load(fj, parse_int=parse_int)
    return model_json


if __name__ == "__main__":
    myz = 0

    # "shape"-字形; "all"-汇总字形/词义/拼音; "w2v"-词义优先+字形; "pinyin"-拼音优先+字形
    # kind = "pinyin"  # "all"  # "w2v"  # "pinyin"  # "shape"
    rounded = 4
    res1 = sim_word("四", "力", code=4, rounded=rounded)
    res2 = sim_word("四", "试", code=4, rounded=rounded)
    print(res1, res2)
    # char1 = "我"
    # char2 = "他"
    # time_start = time.time()
    # res = sim_pinyin(char1, char2, rounded=rounded)
    # time_end = time.time()
    # print(time_end-time_start)
    # print(res)
    # while True:
    #     try:
    #         print("请输入char1: ")
    #         char1 = input()
    #         print("请输入char2: ")
    #         char2 = input()
    #         res = sim_pinyin(char1, char2, rounded=rounded)
    #         print(res)
    #     except Exception as e:
    #         print(traceback.print_exc())
