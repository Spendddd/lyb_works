import json
import os
import time

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_community.output_parsers.ernie_functions import JsonKeyOutputFunctionsParser
from openai import APIConnectionError

from config import environment_config
from infer_task.text_analysis_generation.text_analysis_prompt import plot_analysis_prompt
from infer_task.text_analysis_generation.text_analysis_pydantic import PlotAnalyse
from static_data.tushare_api_mapper import TUSHARE_API_CITATION_MAP

os.environ["http_proxy"] = environment_config['http_proxy']
os.environ["https_proxy"] = environment_config['https_proxy']
os.environ["all_proxy"] = environment_config['all_proxy']

def create_analyse_chain():

    prompt = ChatPromptTemplate.from_messages(plot_analysis_prompt)
    analyse_functions = [convert_pydantic_to_openai_function(PlotAnalyse)]
    model = ChatOpenAI(api_key=environment_config["openai_key"], model=environment_config["summary_model"], temperature=0)

    # 绑定函数描述变量
    analyse_model = model.bind(functions=analyse_functions,
                               function_call={"name": "PlotAnalyse"})

    analyse_chain = prompt | analyse_model | JsonKeyOutputFunctionsParser(key_name="分析")
    return analyse_chain


def run_plot_analyse_chain(plot_data, query_task_type, output_file=None):
    """
    作为文本分析chain的pipeline
        Args:
            query_task_type: 区分个股还是行业分析的任务，给出任务目标名称
            plot_data: 图表数据

            output_file: 图表分析的结果和数据引用
        Returns:
            result:  # todo: 还没修改好
    """
    if not isinstance(plot_data, list):
        with open(plot_data, 'r') as file:
            plot_data = file.read()

    analyse_chain = create_analyse_chain()

    index = 0
    while index < 3:
        index += 1
        try:
            result = analyse_chain.invoke({"input": plot_data})
            print(result)
            break
        except APIConnectionError as e:
            print("无法连接到OpenAI API，请重新尝试")
            time.sleep(30)
        except Exception as e:
            raise e

    plot_analyse = ""
    plot_citation = []

    # 合并分析和citation成plot_analyse的结果
    # 注意：citation的格式是[citation:<2>,<3>,<4>]
    # print(f"plot_data: {plot_data}")
    if len(plot_data) > 0:
        citation_num = 1  # 分析的citation的序号
        citation_text = "[citation:"
        for i in range(len(plot_data)):
            citation_key = f"<{citation_num}>"
            citation_text += citation_key+","
            plot_citation.append({citation_key: TUSHARE_API_CITATION_MAP[plot_data[i][0]]})
            citation_num += 1
        citation_text = citation_text[:-1] + "]"
        plot_analyse += f'{result[:-1]}{citation_text}{result[-1:]}\n'

    plot_analyse_dict = {
        "plot_analyse":
            {
                "analyse": plot_analyse,
                "citation": plot_citation
            },
        "cite_num": len(plot_citation)
    }
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as file:
            json.dump(plot_analyse_dict, file, ensure_ascii=False, indent=4)

    return plot_analyse_dict, len(plot_citation)


if __name__ == "__main__":
    origin_data = {'ids': [['id87', 'id119', 'id70', 'id102']],
                   'distances': [[0.4081319272518158, 0.4081319828368223, 0.562750518321991, 0.5627505549010887]],
                   'metadatas': [[{
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
                           'url': 'https://finance.eastmoney.com/a/202403103007210340.html'},
                       {
                           'source': 'eastmoney',
                           'title': '新能源重磅利好！外资突然唱多宁德时代 黄仁勋、奥特曼：AI的尽头是光伏和储能',
                           'url': 'https://finance.eastmoney.com/a/202403103007210340.html'}]],
                   'embeddings': None, 'documents': [
            ['2024-04-24，据宁德时代微博消息，宁德时代将于4月25日召开2024宁德时代储能新品发布会。',
             '2024-04-24，据宁德时代微博消息，宁德时代将于4月25日召开2024宁德时代储能新品发布会。',
             '2024-03-25，【曾毓群：宁德时代与特斯拉在合作开发充电速度更快的电池】宁德时代董事长曾毓群在接受采访时表示，宁德时代与特斯拉在合作开发充电速度更快的电池。曾毓群透露，双方正共同研究新型电化学结构等电池技术，旨在加快充电速度。曾毓群另外表示，宁德时代正为特斯拉位于美国内华达州的工厂提供设备。',
             '2024-03-25，【曾毓群：宁德时代与特斯拉在合作开发充电速度更快的电池】宁德时代董事长曾毓群在接受采访时表示，宁德时代与特斯拉在合作开发充电速度更快的电池。曾毓群透露，双方正共同研究新型电化学结构等电池技术，旨在加快充电速度。曾毓群另外表示，宁德时代正为特斯拉位于美国内华达州的工厂提供设备。']],
                   'uris': None, 'data': None, 'included': ['metadatas', 'documents', 'distances']}

    plot_analyse_dict, cite_num = run_plot_analyse_chain(plot_data="graph_data.txt", query_task_type="company")
