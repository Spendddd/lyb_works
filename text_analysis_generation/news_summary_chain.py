import json
import os
import time
from typing import List, Optional

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.output_parsers.ernie_functions import JsonKeyOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from openai import APIConnectionError

from config import environment_config
from infer_task.text_analysis_generation.text_analysis_prompt import summary_prompt_template, simply_summary_prompt
from infer_task.text_analysis_generation.text_analysis_pydantic import CompanySummary, IndustrySummary

os.environ["http_proxy"] = environment_config['http_proxy']
os.environ["https_proxy"] = environment_config['https_proxy']
os.environ["all_proxy"] = environment_config['all_proxy']

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return (f"Document(page_content={self.page_content!r}, metadata={self.metadata!r})")

def create_simply_summary_chain(type:str):

    prompt = ChatPromptTemplate.from_messages(simply_summary_prompt)

    # 创建函数描述变量
    if type == "company":
        simply_summary_functions = [convert_pydantic_to_openai_function(CompanySummary)]
    else:
        simply_summary_functions = [convert_pydantic_to_openai_function(IndustrySummary)]

    model = ChatOpenAI(api_key=environment_config["openai_key"], model=environment_config["summary_model"],
                       temperature=0)

    # 绑定函数描述变量
    if type == "company":
        source_extraction_model = model.bind(functions=simply_summary_functions,
                                         function_call={"name": "CompanySummary"})
    else:
        source_extraction_model = model.bind(functions=simply_summary_functions,
                                             function_call={"name": "IndustrySummary"})

    # 创建chain
    source_extraction_chain = prompt | source_extraction_model | JsonKeyOutputFunctionsParser(key_name="总结")

    return source_extraction_chain


def run_news_summary_chain(strategy, query_task, query_task_type, query_data, output_file=None):
    """
        生成新闻总结的文本
        Args:
            strategy:
            query_task: 任务目标名称,可能是company/industry
            query_task_type: 任务目标类型
            query_data: 请求本任务的输入数据
            output_file: summary结果的目录
        Returns:

    """
    llm = ChatOpenAI(api_key=environment_config["openai_key"], model=environment_config["summary_model"], temperature=0)

    # 投资评级分析

    prompt = PromptTemplate.from_template(summary_prompt_template)
    # refine_prompt = PromptTemplate.from_template(refine_template)

    # chain = load_summarize_chain(llm=llm, chain_type='refine', question_prompt=prompt, verbose=False)
    chain = load_summarize_chain(llm=llm,
                                 chain_type='refine',
                                 question_prompt=prompt,
                                 # refine_prompt=refine_prompt,
                                 return_intermediate_steps=False,
                                 input_key="text",
                                 output_key="output_text")

    text_splitter = CharacterTextSplitter(
        separator="。",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False
    )

    # step: reconstruct需要summary的document
    page_content_list = query_data["documents"]
    metadata_list = query_data["metadatas"]

    docs = []
    for page_content, metadata in zip(page_content_list, metadata_list):
        docs.append(Document(page_content[11:], metadata))
    # print(docs)

    split_docs = text_splitter.split_documents(docs)
    print(split_docs)

    simply_summary_chain = create_simply_summary_chain(query_task_type)
    index = 0
    while index < 3:
        index += 1
        try:
            result = chain({"text": split_docs, "query_task": query_task}, return_only_outputs=True)
            break
        except APIConnectionError as e:
            print("无法连接到OpenAI API，请重新尝试")
            time.sleep(30)
        except Exception as e:
            raise e

    output_summary = result["output_text"]
    print(f"output_summary: {output_summary}\n")

    index = 0
    while index < 3:
        index += 1
        try:
            output_summary = simply_summary_chain.invoke({"input": output_summary})
            print(f"simplied_summary: {output_summary}\n")

            break
        except APIConnectionError as e:
            print("无法连接到OpenAI API，请重新尝试")
            time.sleep(30)
        except Exception as e:
            raise e


    if output_file is not None:
        if os.path.exists(output_file):
            with open(output_file, "r", encoding='utf-8') as json_file:
                summary_dict_list = json.load(json_file)
        else:
            summary_dict_list = []
        output = {
            "strategy": strategy,
            "summary": output_summary.replace("\n\n", "")
        }
        print(f"summary_dict: {output}\n")
        summary_dict_list.append(output)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(summary_dict_list, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # query_data = {'ids': [['id20', 'id52', 'id94']], 'distances': [[1.2754310811681475, 1.2754310811681475, 1.2761625112502404]], 'metadatas': [[{'source': 'eastmoney', 'title': '固态电池概念股开盘拉升 三祥新材冲击涨停', 'url': 'https://finance.eastmoney.com/a/202404093037276794.html'}, {'source': 'eastmoney', 'title': '固态电池概念股开盘拉升 三祥新材冲击涨停', 'url': 'https://finance.eastmoney.com/a/202404093037276794.html'}, {'source': 'eastmoney', 'title': '4月9日东方财富财经晚报（附新闻联播）', 'url': 'https://finance.eastmoney.com/a/202404093037917771.html'}]], 'embeddings': None, 'documents': [['2024-05-11，上海钢联发布数据显示，今日电池级碳酸锂价格下跌1500元，均价报11万元/吨。', '2024-05-11，上海钢联发布数据显示，今日电池级碳酸锂价格下跌1500元，均价报11万元/吨。', '2024-06-03，新能源题材午后拉升，晶科能源、许继电气涨超5%，亿纬锂能、比亚迪涨超4%，宁德时代涨超2%。']], 'uris': None, 'data': None, 'included': ['metadatas', 'documents', 'distances']}
    query_data = dict(ids=[['id87', 'id119', 'id70']],
                      distances=[[0.4081319272518158, 0.4081319828368223, 0.562750518321991]],
                      metadatas=[[{
                          'source': 'eastmoney',
                          'title': '固态电池大火 14家券商扎堆推荐宁德时代 新能源车龙头ETF（159637）涨超2%',
                          'url': 'https://finance.eastmoney.com/a/202404093037417958.html'},
                          {
                              'source': 'eastmoney',
                              'title': '固态电池大火 14家券商扎堆推荐宁德时代 新能源车龙头ETF（159637）涨超2%',
                              'url': 'https://finance.eastmoney.com/a/202404093037417958.html'},
                          {
                              'source': 'eastmoney',
                              'title': '新能源重磅利好！外资突然唱多宁德时代 黄仁勋、奥特曼：AI的尽头是光伏和储能',
                              'url': 'https://finance.eastmoney.com/a/202403103007210340.html'}]],
                      embeddings=None,
                      documents=[
                          ['2024-04-24，据宁德时代微博消息，宁德时代将于4月25日召开2024宁德时代储能新品发布会。',
                           '2024-04-24，据宁德时代微博消息，宁德时代将于4月25日召开2024宁德时代储能新品发布会。',
                           '2024-03-25，【曾毓群：宁德时代与特斯拉在合作开发充电速度更快的电池】宁德时代董事长曾毓群在接受采访时表示，宁德时代与特斯拉在合作开发充电速度更快的电池。曾毓群透露，双方正共同研究新型电化学结构等电池技术，旨在加快充电速度。曾毓群另外表示，宁德时代正为特斯拉位于美国内华达州的工厂提供设备。']],
                      uris=None, data=None, included=['metadatas', 'documents', 'distances'])
    strategy = "summary20"
    run_news_summary_chain(
        strategy=strategy,
        query_task="宁德时代",
        query_task_type="company",
        query_data=query_data,
        output_file='ningde_summary.json')
