import json
import os
import re
import time
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.output_parsers.ernie_functions import JsonKeyOutputFunctionsParser
from openai import APIConnectionError

from config import environment_config
from infer_task.text_analysis_generation.text_analysis_prompt import source_extract_prompt
from infer_task.text_analysis_generation.text_analysis_pydantic import SourceInfo

os.environ["http_proxy"] = environment_config['http_proxy']
os.environ["https_proxy"] = environment_config['https_proxy']
os.environ["all_proxy"] = environment_config['all_proxy']


# Extract the relevant information, if not explicitly provided do not guess.

def create_source_extraction_chain():
    convert_pydantic_to_openai_function(SourceInfo)

    prompt = ChatPromptTemplate.from_messages(source_extract_prompt)

    # 创建函数描述变量
    source_extraction_functions = [convert_pydantic_to_openai_function(SourceInfo)]
    model = ChatOpenAI(api_key=environment_config["openai_key"], model=environment_config["summary_model"],
                       temperature=0)

    # 绑定函数描述变量
    source_extraction_model = model.bind(functions=source_extraction_functions,
                                         function_call={"name": "SourceInfo"})

    # 创建chain
    source_extraction_chain = prompt | source_extraction_model | JsonKeyOutputFunctionsParser(key_name="来源")

    return source_extraction_chain


# query_plot_analysis_file: 图表分析的结果和数据引用
# query_news_summary_file: summary结果的文件路径
# query_news_citation_file: 引用结果的文件路径
# query_analysis_text_file: 完整的analysis_text结果的文件路径

def run_source_extraction_merge_plot_analysis(strategy, query_task_type, query_plot_analysis_file, query_news_citation_file,
                                              query_news_summary_file, output_file):
    """
        作为文本分析chain的pipeline
        Args:
            query_task_type: "company" or "industry"
            strategy:
            query_plot_analysis_file:图表分析的文件路径
            query_news_summary_file: summary结果的文件路径
            query_news_citation_file: 引用结果的文件路径
            output_file: 完整的analysis_text结果的文件路径

        Returns:
            result:
    """
    path = Path(query_plot_analysis_file)
    if path.exists():
        with open(query_plot_analysis_file, 'r', encoding='utf-8') as json_file:
            plot_analyse_dict = json.load(json_file)
            cite_num = plot_analyse_dict["cite_num"]
    else:
        plot_analyse_dict = {}
        cite_num = 0

    with open(query_news_citation_file, 'r', encoding='utf-8') as citation_file, \
            open(query_news_summary_file, 'r', encoding='utf-8') as summary_file:
        citation_dict_list = json.load(citation_file)
        summary_dict_list = json.load(summary_file)
    citation_dict = {}
    summary_dict = {}
    for citation in citation_dict_list:
        if citation["strategy"] == strategy:
            citation_dict = citation
    for summary in summary_dict_list:
        if summary["strategy"] == strategy:
            summary_dict = summary

    if not citation_dict or not summary_dict:
        return None

    # source_extraction_chain = create_source_extraction_chain()

    if os.path.exists(output_file):
        with open(output_file, "r", encoding='utf-8') as json_file:
            text_analysis_dict_list = json.load(json_file)
    else:
        text_analysis_dict_list = []

    # [citation: <5>]
    # 包括新总结文本和引用字典列表的字典
    text_analysis_dict = {
        "strategy": strategy
    }

    current_cite_num = cite_num + 1
    citation_with_source_list = []
    citation_isin_list = []

    summary = summary_dict["summary"]
    for citation in citation_dict["citation_list"]:
        # # 创建loader,获取网页数据
        # loader = WebBaseLoader(citation["url"])
        # documents = loader.load()
        # # 查看网页内容
        # doc = documents[0]
        # # 获取来源
        # index = 0
        # while index < 3:
        #     index += 1
        #     try:
        #         source_dict_result = source_extraction_chain.invoke({"input": doc})
        #         print(f"source_dict_result: {source_dict_result}")
        #         break
        #     except APIConnectionError as e:
        #         print("无法连接到OpenAI API，请重新尝试")
        #         time.sleep(30)
        #     except Exception as e:
        #         raise e

        # if source_dict_result:
        #     new_citation = f"{citation['来源']}-{source_dict_result}"
        # else:
        #     new_citation = f"{citation['来源']}"
        new_citation = f"{citation['来源']}-{citation['url']}"

        if new_citation not in citation_isin_list:
            citation_key = f"<{current_cite_num}>"
            current_cite_num += 1
            citation_isin_list.append(new_citation)
            citation_with_source_list.append({citation_key: new_citation})
            if citation["总结文本片段"].endswith("，") or citation["总结文本片段"].endswith("。"):
                summary = summary.replace(citation["总结文本片段"],
                                          f'{citation["总结文本片段"][:-1]}[citation:{citation_key}]{citation["总结文本片段"][-1:]}')
            else:
                summary = summary.replace(citation["总结文本片段"], f'{citation["总结文本片段"]}[citation:{citation_key}]')

        else:
            for i in range(len(citation_isin_list)):
                if citation_isin_list[i] == new_citation:
                    citation_key = f"<{cite_num + i + 1}>"
                    if citation["总结文本片段"].endswith("，") or citation["总结文本片段"].endswith("。"):
                        summary = summary.replace(citation["总结文本片段"],
                                                  f'{citation["总结文本片段"][:-1]}[citation:{citation_key}]{citation["总结文本片段"][-1:]}')
                    else:
                        summary = summary.replace(citation["总结文本片段"], f'{citation["总结文本片段"]}[citation:{citation_key}]')
                    break

    print(f"citation_with_source_list: {citation_with_source_list}")
    if len(plot_analyse_dict) > 0:
        if len(summary) > 300 - len(plot_analyse_dict["plot_analyse"]["analyse"]) and query_task_type == "company":
            summary = process_text(summary, 300 - len(plot_analyse_dict["plot_analyse"]["analyse"]),
                                   200 - len(plot_analyse_dict["plot_analyse"]["analyse"]))
        elif len(summary) > 500 - len(plot_analyse_dict["plot_analyse"]["analyse"]) and query_task_type == "industry":
            summary = process_text(summary, 500 - len(plot_analyse_dict["plot_analyse"]["analyse"]),
                                   300 - len(plot_analyse_dict["plot_analyse"]["analyse"]))

        print(f"plot_analyse_dict: {plot_analyse_dict}")

        text_analysis_dict["summary_text"] = plot_analyse_dict["plot_analyse"]["analyse"] + summary

        citations = re.findall(r'<[^>]+>', text_analysis_dict["summary_text"])

        chunk_citation_list = plot_analyse_dict["plot_analyse"]["citation"]
        if citations:
            last_cite = citations[-1]
            for i in range(len(citation_with_source_list)):
                chunk_citation_list.append(citation_with_source_list[i])
                if last_cite in citation_with_source_list[i].keys():
                    break
        citation_with_source_list = chunk_citation_list
        print(citation_with_source_list)

    text_analysis_dict["citation"] = citation_with_source_list
    print(f"text_analysis_dict: {text_analysis_dict}\n")
    text_analysis_dict_list.append(text_analysis_dict)

    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(text_analysis_dict_list, json_file, ensure_ascii=False, indent=4)

    return text_analysis_dict


def process_text(input_text, limit_len, min_len):
    # 将文本按句子分割
    sentences = input_text.split('。')
    result = []
    current_length = 0
    now_text = ""

    for sentence in sentences:
        # 保留句号作为结束符
        sentence += '。'
        # 计算当前文本加上这个句子的长度
        new_length = current_length + len(sentence)

        if new_length <= limit_len:
            # 如果加上这句话小于限制长度，加上这句话
            now_text += sentence
            current_length = new_length
        elif current_length < min_len:
            # 如果当前长度小于最小长度，则分割成短句处理
            short_sentences = sentence.split('，')
            for i in range(len(short_sentences)):
                short_sentences[i] += '，'
                sentence = ""
                if i > 0:
                    sentence.replace(short_sentences[i-1]+"，", "")
                new_length = current_length + len(short_sentences[i])

                if new_length > limit_len:
                    # 如果加上这个短句超过限制长度，跳过此短句并停止检查
                    break
                # 否则添加这个短句到结果
                if not short_sentences[i][:-1].endswith(">]") and "[citation:" in sentence:
                    pattern = r'\[citation:[^\]]+\]'
                    citations = re.findall(pattern, sentence)
                    short_sentence = short_sentences[i] + citations[0]
                    if current_length + len(short_sentence) > limit_len:
                        break
                now_text += short_sentences[i]
                current_length += len(short_sentences[i])
            if "[citation:" in sentence and not now_text[:-1].endswith(">]"):
                pattern = r'\[citation:[^\]]+\]'
                citations = re.findall(pattern, sentence)
                now_text = now_text[:-1] + citations[0] + "。"
            else:
                now_text = now_text[:-1] + "。"
            # 停止对当前句子的进一步检查
            break
        else:
            break
    if now_text.endswith("。。"):
        now_text = now_text[:-1]
    return now_text


if __name__ == "__main__":
    run_source_extraction_merge_plot_analysis(
        plot_analyse_dict="../summary_generation/plot_analyse.json",
        cite_num=2,
        query_task_type="company",
        query_news_citation_file='../summary_generation/ningde_result.json',
        query_news_summary_file='../summary_generation/ningde_summary.json',
        output_dir='../summary_generation/ningde_full_citation.json')
