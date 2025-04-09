import json
import os
import re
import time
import logging

import jieba
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langchain_community.output_parsers.ernie_functions import JsonKeyOutputFunctionsParser
from langchain_openai import ChatOpenAI
from openai import APIConnectionError
from pydantic import BaseModel, Field

from .char_similar.pinyin_only import find_sim_word_in_list
# from char_similar.pinyin_only import find_sim_word_in_list

# 指定整个代码文件使用的openai的api_key
# os.environ[
#     'OPENAI_API_KEY'] = 'sk-*****'
"""
定义处理分词、搜索相近词、替换热词、语义纠正任务的类，
初始化参数：
    model：调用openai的模型名称，默认值为gpt-4
    temperature：模型输出随机性，默认为0
"""

logger = logging.getLogger(__name__)

class Replacemant(BaseModel):
    """对输入的句子进行不文明用语处理和热词替换处理"""
    处理后句子: str = Field(descrpition="完成不文明用语处理和热词替换后的句子")

class ReplacementList(BaseModel):
    """整合完成不文明用语处理和热词替换后的句子为列表"""
    处理后句子列表: list[Replacemant] = Field(descrpition="完成不文明用语处理和热词替换后的完整句子列表")

def create_replacement_chain(replacement_prompt, model):

    prompt = ChatPromptTemplate.from_messages(replacement_prompt)

    # 创建函数描述变量
    replacement_functions = [convert_pydantic_to_openai_function(ReplacementList)]

    # 绑定函数描述变量
    replacement_model = model.bind(functions=replacement_functions,
                                  function_call={"name": "ReplacementList"})

    # 创建chain
    replacement_chain = prompt | replacement_model | JsonKeyOutputFunctionsParser(key_name="处理后句子列表")

    return replacement_chain

def audio_content_processing(sentence_list, config, service_config):
    """
    需要的配置信息：
        hotwords_dir：热词列表文件（.txt）的位置【必须】
        badwords_dir：负向词列表文件（.txt）的位置【必须】
        model：指定gpt的模型版本，默认为“gpt-3.5-turbo”【非必须】
        temperature：指定gpt的生成结果随机性，默认为0【非必须】
        max_score：混合热词词典中与热词的最低相似程度（分数越高越不相似），默认为4【非必须】
        limit：单个热词词典每个热词的相似词列表的最大长度，默认为20【非必须】
        per_len：模型每次接受的最大content数，默认为25【非必须】
    """
    model = service_config["senior_model"]
    temperature = config["temperature"]
    process = PostProcess_hotwords_replace(api_key=service_config['openai_key'],
                                           model=model,
                                           temperature=temperature)
    max_score = config["max_score"]
    limit = config["limit"]
    per_len = config["per_len"]
    box_size = config["box_size"]
    percent = config["percent"]
    window_size = config["window_size"]
    replace_dir = config["replace_dir"]
    hotwords_dir = config["hotwords_dir"]
    badwords_dir = config["badwords_dir"]
    bad_replace_dir = config["bad_replace_dir"]
    # cur_dir = os.path.dirname(__file__)
    # replace_dir = os.path.join(cur_dir, 'replace.json')
    return process.run_hotwords_replacement(sentence_list=sentence_list,
                                            hotwords_dir=hotwords_dir,
                                            badwords_dir=badwords_dir,
                                            max_score=max_score,
                                            replace_dir=replace_dir,
                                            bad_replace_dir=bad_replace_dir,
                                            limit=limit,
                                            per_len=per_len,
                                            box_size=box_size,
                                            percent=percent,
                                            window_size=window_size)


class PostProcess_hotwords_replace:

    def __init__(self, api_key, model="gpt-3.5-turbo", temperature=0):
        self.llm = ChatOpenAI(api_key=api_key,
                              model=model,
                              temperature=temperature)
        self.tagging_model = ChatOpenAI(api_key=api_key,
                                        model=model,
                                        temperature=temperature)

    def run_hotwords_replacement(self,
                                 sentence_list,
                                 hotwords_dir,
                                 badwords_dir,
                                 max_score=4,
                                 replace_dir=None,
                                 bad_replace_dir=None,
                                 limit=20,
                                 per_len=25,
                                 box_size=50,
                                 percent=0.5,
                                 window_size=3
                                 ):
        # 获取n-gram和jieba分词对应的拼音相似度检索结果
        hotwords_dict_ngram, badwords_dict_ngram = self.run_search_by_only_pinyin(
            sentence_list,
            hotwords_dir,
            badwords_dir,
            replace_dir=replace_dir,
            bad_replace_dir=bad_replace_dir,
            limit=limit,
            cut_type="n-gram")
        logger.info("dialog n-gram热词列表、负向词列表生成完成. ")

        hotwords_dict_jieba, badwords_dict_jieba = self.run_search_by_only_pinyin(
            sentence_list,
            hotwords_dir,
            badwords_dir,
            replace_dir=replace_dir,
            bad_replace_dir=bad_replace_dir,
            limit=limit,
            cut_type="jieba")
        logger.info("dialog jieba热词列表、负向词列表生成完成. ")

        # 生成混合分词热词字典
        hotwords_dict_multi2 = self.get_multi_words_dict(hotwords_dict_ngram, hotwords_dict_jieba,
                                                         max_score, hotwords_dir=hotwords_dir, badwords_dir=None)
        print(f"混合热词列表：{hotwords_dict_multi2}")
        logger.info("dialog 混合热词列表生成完成：%s ", json.dumps(hotwords_dict_multi2))

        badwords_dict_multi2 = self.get_multi_words_dict(badwords_dict_ngram, badwords_dict_jieba,
                                                         max_score=2, hotwords_dir=None, badwords_dir=badwords_dir)

        print(f"混合负向词列表：{badwords_dict_multi2}")
        logger.info("dialog 混合负向词列表生成完成：%s ", json.dumps(badwords_dict_multi2))

        # 正则检索待进行替换处理的句子，调用gpt完成热词替换
        return self.replace_with_re_then_chat(self.tagging_model, sentence_list, hotwords_dict_multi2, badwords_dict_multi2,
                                              max_score, per_len, box_size, percent, window_size)

    """通过正则的方式利用混合分词热词字典获取待进行替换处理的content，调用gpt完成替换"""

    def replace_with_re_then_chat(self,
                                  tagging_model,
                                  sentence_list,
                                  hotwords_dict,
                                  badwords_dict,
                                  max_score=4,
                                  per_len=25,
                                  box_size=50,
                                  percent=0.5,
                                  window_size=3):

        logger.info("dialog 开始执行热词和负向词替换. ")
        output = []
        result_list = self.preprocess_with_role(sentence_list)
        monitor_data = {}
        monitor_data["热词词典"] = hotwords_dict
        monitor_data["负向词词典"] = badwords_dict
        holding_replace_num = 0
        monitor_data["候选句子列表"] = []
        monitor_data["候选热词列表"] = []
        monitor_data["候选负向词列表"] = []

        # 指定gpt顾及语义利用非完整热词列表进行替换
        system_prompt = [("system", "这是一段由钻戒门店中导购与客户关于咨询购买钻戒的对话录音转写的文本中的部分句子，"
                                    "在这些句子中可能存在转写错误的情况。"
                                    "请在保证语义完整的前提下，将在对话文本中出现的与热词读音相近的文本替换成对应的热词，"
                                    "并对文本中出现的不文明用语以及与不文明用语相近的词语或短语替换成相应字数的“*”，并输出执行完这两项任务后的结果。"
                                    "注意先执行热词替换再执行不文明用语替换，另外不要对原始文本进行任何其他操作，对话文本中的每句话以换行符进行划分。"
                                    "不文明用语列表如下：{badwords}，热词列表如下：{hotwords}。对话文本如下："),
                         ("human", "{input}")]
        replacement_chain = create_replacement_chain(system_prompt, tagging_model)
        hotwords_has_words = []
        hotwords_isin = []

        st_idx = 0
        while st_idx < len(result_list):
            end_idx = min(len(result_list), st_idx + box_size)

            # 获取从st_idx开始到end_idx的元素
            current_box = result_list[st_idx:end_idx]
            current_box_idx = list(range(st_idx, end_idx))

            has_words = []
            isin = []
            # 非完整版热词
            hotwords_incomplete = []
            badwords_incomplete = []
            for inx, sentence in zip(current_box_idx, current_box):
                flag = False
                for key, values in hotwords_dict.items():
                    for value in values:
                        if value in sentence:
                            flag = True
                            if key not in hotwords_incomplete:
                                hotwords_incomplete.append(key)
                bad_flag = False
                for key, values in badwords_dict.items():
                    for value in values:
                        if value in sentence:
                            bad_flag = True
                            if key not in badwords_incomplete:
                                badwords_incomplete.append(key)
                if flag or bad_flag:
                    has_words.append(sentence)
                    isin.append(inx)
                    if flag:
                        hotwords_isin.append(inx)
                        hotwords_has_words.append(sentence)
                    print(sentence)
            print(isin)

            if len(isin)/len(current_box_idx) < percent:
                # 候选句子的前后三句都要加入候选句子
                new_isin = []
                new_has_words = []
                for idx in isin:
                    for i in range(-window_size, window_size + 1):
                        if idx + i >= st_idx and idx + i < end_idx and idx + i not in new_isin:
                            new_isin.append(idx + i)
                            new_has_words.append(result_list[idx + i])
                isin = new_isin
                has_words = new_has_words
            print(f"isin: {isin}")
            logger.info("dialog 当前待处理的句子下标：%s ", json.dumps(isin))
            print(f"has_words: {has_words}\n")
            logger.info("dialog 当前待处理的句子：%s ", json.dumps(has_words))

            holding_replace_num += len(isin)
            monitor_data["候选句子列表"].extend(has_words)
            print(hotwords_incomplete)
            print(badwords_incomplete)
            logger.info("dialog 当前待传入gpt的热词列表：%s ", json.dumps(hotwords_incomplete))
            logger.info("dialog 当前待传入gpt的负向词列表：%s ", json.dumps(badwords_incomplete))
            monitor_data["候选热词列表"].extend(hotwords_incomplete)
            monitor_data["候选负向词列表"].extend(badwords_incomplete)
            monitor_data["候选热词列表"] = list(set(monitor_data["候选热词列表"]))
            monitor_data["候选负向词列表"] = list(set(monitor_data["候选负向词列表"]))

            # .format(hotwords=hotwords_incomplete))
            # 按以下格式输出：每句一行，角色：文本，不要输出空行
            process_result = []
            # 初始化索引
            start_index = 0
            # 循环处理待处理句子
            while start_index < len(has_words):
                end_index = min(len(has_words), start_index + per_len)

                # 获取从start_index开始到end_index的元素
                current_slice = has_words[start_index:end_index]

                # 将这些元素拼接成字符串
                concatenated_string = '\n'.join(current_slice)
                while 1:
                    try:
                        replacement_list = replacement_chain.invoke({"input": concatenated_string,
                                                                     "hotwords": hotwords_incomplete,
                                                                     "badwords": badwords_incomplete})
                        break
                    except APIConnectionError as e:
                        print("无法连接到OpenAI API，请重新尝试")
                        time.sleep(30)
                        logger.error("无法连接到OpenAI API，请重新尝试")
                    except Exception as e:
                        logger.error(e)
                        # raise e

                process_result.extend(replacement_list)
                print(f"process_result: {process_result}")
                logger.info("dialog 当前处理轮次返回结果：%s ", json.dumps(process_result))

                if start_index + per_len < len(has_words):
                    start_index += per_len
                else:
                    break  # 退出循环，因为已处理所有元素

            j = 0
            isin_index = []
            for sentence in process_result:
                temp_sentence = sentence["处理后句子"].split(':')
                temp = temp_sentence[1].strip() if len(temp_sentence) > 1 else temp_sentence[0].strip()
                if (sentence_list[isin[j]]['sentence_id'] not in isin_index
                        and temp != sentence_list[isin[j]]['content']):
                    add_data = {
                        'sentence_id': sentence_list[isin[j]]['sentence_id'],
                        'content': temp
                    }
                    output.append(add_data)
                    isin_index.append(sentence_list[isin[j]]['sentence_id'])
                j += 1
            print(output)
            logger.info("dialog 当前箱完成后累计处理结果：%s ", json.dumps(output))

            if st_idx - window_size + box_size < len(result_list):
                st_idx += box_size - window_size
            else:
                break  # 退出循环，因为已处理所有元素


        monitor_data["候选句子数"] = f"{holding_replace_num}"
        monitor_data["候选含热词句子数"] = f"{len(hotwords_has_words)}"
        monitor_data["候选句子占比"] = f"{(holding_replace_num / len(result_list)):.2%}"
        monitor_data["候选含热词句子占比"] = f"{(len(hotwords_has_words) / len(result_list)):.2%}"
        if len(output) > 0:
            print(output)
            monitor_data["执行替换句子数"] = f"{len(output)}"
            monitor_data[
                "执行替换句子占候选句子比例"] = f"{(len(output)/holding_replace_num):.2%}"
            monitor_data[
                "执行替换句子占全部句子比例"] = f"{(len(output)/len(result_list)):.2%}"
            monitor_data["执行替换句子列表"] = output
        with open("monitor_data" + str(max_score) + "_test.json",
                  "w",
                  encoding='utf-8') as json_file:
            json.dump(monitor_data, json_file, ensure_ascii=False, indent=4)
        logger.info("dialog 热词替换统计指标：%s ", json.dumps(monitor_data))
        logger.info("dialog 热词替换完成。 ")

        return output




    """语义纠正"""

    def run_correct(self, indir, outdir, per_len=25):
        # 获取原始输入的分句结果（列表）
        result_list = []
        with open(indir + ".txt", "r", encoding='utf-8') as file:
            for line in file:
                result_list.append(line)

        system_prompt_correct = (
            "这是一段钻戒门店中导购与客户关于咨询购买钻戒的对话录音转写成的文本结果。对话参与人员配置如下，其他情况以此类推："
            "如果有2个参与者，则设置为1位导购和1位客户；"
            "如果有3个参与者，则设置为1位导购和1对情侣或夫妻客户，或2位导购和1位客户；"
            "如果有4个参与者，则设置为2位导购和2位客户。"
            "在这段对话文本中，可能存在由于识别转写效果不佳导致的语义错误、非标准表达和不合逻辑的表达，请尽量识别出这些错误并进行纠正，使得对话的语义更连贯流畅，并且更贴近此类销售场景中的真人对话。"
            "根据你对钻戒知识的了解，纠正可能被转写错误的词语和短语。"
            "这段对话可能包含方言，当识别到语义不连贯或不通顺的文本时，请考虑将此处标记为【方言】。"
            "注意不要对原始文本进行任何概括性的转写，按以下格式输出：每句一行，角色：文本，不要输出空行。"
            "对话文本如下：")
        process_result = ""
        # 初始化索引
        start_index = 0

        # 循环，直到没有足够的元素形成新的组合
        while start_index < len(result_list):
            end_index = min(len(result_list), start_index + per_len)

            # 获取从start_index开始到end_index的元素
            current_slice = result_list[start_index:end_index]

            # 将这些元素拼接成字符串
            concatenated_string = ' '.join(current_slice)
            content = self.complete_chat(concatenated_string,
                                         system_prompt_correct)
            cleaned_content = re.sub(r'\n\s*\n', '\n', content)
            content = cleaned_content.strip()
            process_result += content

            if start_index + per_len < len(result_list):
                start_index += per_len
            else:
                break  # 退出循环，因为已处理所有元素

        with open(outdir + "_origin.txt", "w", encoding='utf-8') as file:
            file.write(process_result)

        parts = re.split(r'(?=\d+:)', process_result)
        parts = [part.strip() for part in parts if part.strip()]
        formatted_lines = [part for part in parts]
        process_result = '\n'.join(formatted_lines)

        with open(outdir + ".txt", "w", encoding='utf-8') as file:
            file.write(process_result)

    """根据n-gram和jieba分词的结果获取混合分词结果"""

    def get_multi_words_dict(self, words_dict_ngram, words_dict_jieba, max_score,
                             hotwords_dir=None, badwords_dir=None):
        # 取出score <= max_score的词进行正则替换
        words_dir = hotwords_dir if hotwords_dir is not None else badwords_dir
        words = self.get_words(words_dir)
        words_dict = {word: [] for word in words}

        for key, word_list in words_dict_ngram.items():
            # 找出score <= max_score的word
            filtered_words = [
                word_entry['word']
                for word_entry in word_list
                if int(word_entry['score']) <= max_score and
                word_entry['word'] not in words
            ]
            if filtered_words:
                words_dict[key].extend(filtered_words)

        for key, word_list in words_dict_jieba.items():
            # 找出score <= max_score的word
            filtered_words = [
                word_entry['word']
                for word_entry in word_list
                if int(word_entry['score']) <= max_score and
                word_entry['word'] not in words
            ]
            if filtered_words:
                words_dict[key].extend(filtered_words)

        empty_key = []
        for key, word_list in words_dict.items():
            if not word_list:
                empty_key.append(key)
            words_dict[key] = list(set(word_list))

        for key in empty_key:
            del words_dict[key]

        if hotwords_dir is not None and "DR" not in words_dict:
            words_dict["DR"] = []
        if badwords_dir is not None:
            for key, temp_words in words_dict.items():
                temp_words.append(key)
        return words_dict

    """逐句进行分词+拼音相似度检索的工作"""

    def run_search_by_only_pinyin(self,
                                  sentence_list,
                                  hotwords_dir,
                                  badwords_dir,
                                  replace_dir=None,
                                  bad_replace_dir=None,
                                  limit=20,
                                  cut_type="n-gram",
                                  code=4,
                                  rounded=4):
        # 获取原始输入的分句结果（列表）
        result_list = self.preprocess_to_no_sig(sentence_list)
        # print(result_list)
        hotwords = self.get_words(hotwords_dir)
        badwords = self.get_words(badwords_dir)
        n_list = []
        for word in hotwords:
            n_list.append(len(word))
        for word in badwords:
            n_list.append(len(word))
        n_list = list(set(n_list))

        word_list = []
        for text in result_list:
            if cut_type == "n-gram":
                for n in n_list:
                    word_list.extend(self.char_ngram(text, n))
            elif cut_type == "jieba":
                word_list.extend(self.jieba_cut_words(text))

        word_list = list(set(word_list))
        # print(word_list)
        query_list = hotwords
        bad_query_list = badwords
        # print(query_list)

        return (self.search_familior_words_only_pinyin(word_list, query_list,
                                                      limit, replace_dir, code,
                                                      rounded),
                self.search_familior_words_only_pinyin(word_list, bad_query_list,
                                                      limit, bad_replace_dir, code,
                                                      rounded))

    """调用拼音相似度评分代码进行相似度检索"""

    def search_familior_words_only_pinyin(self,
                                          word_list,
                                          query_list,
                                          limit,
                                          replace_dir=None,
                                          code=4,
                                          rounded=4):
        data = {}
        replace_dict = {}
        if replace_dir:
            with open(replace_dir, "r", encoding='utf-8') as json_file:
                replace_dict = json.load(json_file)
        for word in query_list:
            if replace_dict and word in replace_dict:
                holding_list = []
                for holding_word in replace_dict[word]:
                    holding_list.extend(
                        find_sim_word_in_list(word_list,
                                              holding_word,
                                              limit,
                                              code=code,
                                              rounded=rounded))
                data[word] = holding_list
            else:
                data[word] = find_sim_word_in_list(word_list,
                                                      word,
                                                      limit,
                                                      code=code,
                                                      rounded=rounded)

        print(f"词语列表：{data}")
        return data

    """jiaba分词代码"""

    def jieba_cut_words(self, text):
        results = jieba.cut(text, cut_all=True)
        print(results)
        return results

    """n-gram分词代码"""

    def char_ngram(self, text, n):
        # 使用 zip 函数生成字符n-gram
        ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
        print(ngrams)
        return ngrams

    """通过langchain完成chat调用"""

    def complete_chat(self, process_result, system_prompt):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=process_result),
        ]
        return self.llm.invoke(messages).content

    """逐句获取不带标点符号的input_txt的content，以列表形式输出"""

    def preprocess_to_no_sig(self, sentence_list):
        # 遍历 sentence_list 并格式化输出
        result = []
        for sentence in sentence_list:
            # formatted_sentence = f"{sentence['content']}"
            formatted_sentence = re.sub(r'[：、，。:？ ,.?\'"“”‘’]', '',
                                        sentence['content'])
            result.append(formatted_sentence)
        return result

    def preprocess_with_role(self, sentence_list):
        # 遍历 sentence_list 并格式化输出
        result = []
        for sentence in sentence_list:
            formatted_sentence = f"{sentence['role']}: {sentence['content']}"
            result.append(formatted_sentence)

        return result

    """读取热词列表文件，以列表形式输出热词列表"""

    def get_words(self, words_dir):
        lines_list = []

        # 当前文件夹
        words_dir = os.path.join(os.path.dirname(__file__), words_dir)
        # 打开文件并读取每一行
        with open(words_dir, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行末的换行符
                cleaned_line = line.strip()
                lines_list.append(cleaned_line)

        return lines_list


if __name__ == '__main__':
    with open("input_service_sentence_list.json", "r") as f:
        service_dict = json.load(f)
    config = {
        "hotwords_dir": "hotwords.txt",
        "badwords_dir": "badwords.txt",
        "replace_dir": "replace.json",
        "bad_replace_dir": "bad_replace.json",
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_score": 4,
        "limit": 20,
        "per_len": 25,
        "box_size": 50,
        "window_size": 3,
        "percent": 0.5
    }
    service_config = {"openai_key": "sk-***"}
    for service_list in service_dict["service_list"]:
        service_sentence_list = service_list["service_sentence_list"]
        content_result = audio_content_processing(service_sentence_list,
                                                  config,
                                                  service_config)
        with open("result.json", "w", encoding='utf-8') as json_file:
            json.dump(content_result, json_file, ensure_ascii=False, indent=4)
    # with open('input_text.json', 'r', encoding='utf-8') as json_file:
    #     sentence_list = json.load(json_file)["sentence_list"]
    # config = {
    #     'hotwords_dir': "hotwords.txt",
    #     'replace_dir': "replace.json",
    #     'model': "gpt-4",
    #     'temperature': 0,
    #     'max_score': 3,
    #     'limit': 20,
    #     'per_len': 25
    # }
    # output = audio_content_processing(sentence_list, config)
    # print(output)
    #
    # config = {
    #     'hotwords_dir': "hotwords.txt",
    #     'replace_dir': "replace.json",
    #     'model': "gpt-4",
    #     'temperature': 0,
    #     'max_score': 4,
    #     'limit': 20,
    #     'per_len': 25
    # }
    # output = audio_content_processing(sentence_list, config)
    # print(output)
    # config = {
    #     'hotwords_dir': "hotwords.txt",
    #     'replace_dir': "replace.json",
    #     'model': "gpt-4",
    #     'temperature': 0,
    #     'max_score': 3,
    #     'limit': 20,
    #     'per_len': 25
    # }
    # output = audio_content_processing(sentence_list, config)
    # print(output)
    # config = {
    #     'hotwords_dir': "hotwords.txt",
    #     'replace_dir': "replace.json",
    #     'model': "gpt-4",
    #     'temperature': 0,
    #     'max_score': 2,
    #     'limit': 20,
    #     'per_len': 25
    # }
    # output = audio_content_processing(sentence_list, config)
    # print(output)
