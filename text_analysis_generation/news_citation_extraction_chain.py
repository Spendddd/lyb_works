import json
import os
import re
import time
from typing import List, Optional

import openai
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.output_parsers.ernie_functions import JsonKeyOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from openai import APIConnectionError

from config import environment_config, data_config
from infer_task.text_analysis_generation.text_analysis_prompt import news_citation_prompt
from infer_task.text_analysis_generation.text_analysis_pydantic import CitationList

os.environ["http_proxy"] = environment_config['http_proxy']
os.environ["https_proxy"] = environment_config['https_proxy']
os.environ["all_proxy"] = environment_config['all_proxy']
# "注意被标注的文本应直接从总结文本中抽取，并且被抽取的文本按顺序组合应与给出的总结文本一致。"


def create_citation_extraction_chain():
    convert_pydantic_to_openai_function(CitationList)

    prompt = ChatPromptTemplate.from_messages(news_citation_prompt)

    # 创建函数描述变量
    extraction_functions = [convert_pydantic_to_openai_function(CitationList)]
    model = ChatOpenAI(api_key=environment_config["openai_key"], model=environment_config["summary_model"], temperature=0)

    # 绑定函数描述变量
    extraction_model = model.bind(functions=extraction_functions,
                                  function_call={"name": "CitationList"})

    # 创建chain
    extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="citation_list")

    return extraction_chain


def run_citation_extraction_chain(strategy, query_data, query_news_summary_file, output_file):
    """
        提取citation
        Args:
            strategy:
            query_data: 请求本任务的输入数据
            query_news_summary_file：新闻总结的中间文件
            output_file: summary结果的目录
        Returns:

    """
    extraction_chain = create_citation_extraction_chain()
    source_list = []

    page_content_list = query_data["documents"]
    metadata_list = query_data["metadatas"]
    news_data_list = []

    for page_content, metadata in zip(page_content_list, metadata_list):
        source = {
            "标题": metadata["title"],
            "参考原文": page_content[11:],
        }
        news_data = {
            "url": metadata["url"],
            "title": metadata["title"],
            "original_text": page_content[11:]
        }
        source_list.append(source)
        news_data_list.append(news_data)

    with open(query_news_summary_file, "r", encoding='utf-8') as json_file:
        summary_dict_list = json.load(json_file)

    summary_dict = {}
    for summary in summary_dict_list:
        if summary["strategy"] == strategy:
            summary_dict = summary
            break

    if os.path.exists(output_file):
        with open(output_file, "r", encoding='utf-8') as json_file:
            citation_dict_list = json.load(json_file)
    else:
        citation_dict_list = []

    citation_dict = {}
    retry_flag = True
    while retry_flag:
        # llm调用
        index = 0
        while index < 3:
            index += 1
            try:
                citation_list_result = extraction_chain.invoke({"input": summary_dict["summary"], "source": source_list})
                print(f"citation_list_result: {citation_list_result}\n")
                break
            except APIConnectionError as e:
                print("无法连接到OpenAI API，请重新尝试")
                time.sleep(30)
            except Exception as e:
                raise e

        for citation in citation_list_result:
            if citation["来源"] == "总结文本":
                retry_flag = True
                break
            retry_flag = False
        if not retry_flag:
            for citation in citation_list_result:
                url = ""
                for news_data in news_data_list:
                    if (citation["来源"] in news_data["original_text"]
                            or citation["来源"] in news_data["title"]):
                        url = news_data["url"]
                        citation["来源"] = news_data["title"]
                        break
                citation["url"] = url
            citation_dict["strategy"] = strategy
            citation_dict["citation_list"] = citation_list_result

    citation_dict_list.append(citation_dict)

    if output_file is not None:
        with open(output_file, "w", encoding='utf-8') as json_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            json.dump(citation_dict_list, json_file, ensure_ascii=False, indent=4)
            # print(result.additional_kwargs['function_call']['arguments'])
