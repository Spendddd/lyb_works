import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import data_config
from data.get_news_from_api import get_news_from_api
from infer_task.text_analysis_generation.news_citation_complete_merge_plot_analysis import \
    run_source_extraction_merge_plot_analysis
from infer_task.text_analysis_generation.news_citation_extraction_chain import run_citation_extraction_chain

from infer_task.text_analysis_generation.plot_analysis_chain import run_plot_analyse_chain
from infer_task.text_analysis_generation.news_summary_chain import run_news_summary_chain
from static_data.query_task_mapper import task1_A_company, task2_A_industry
from storage.chroma_operator import VectorDBConnector
from utils.utils import filter_news_from_query_data, get_start_end_date


def run_analysis_text_pipeline(strategy:str,
                               query_task: str,
                               query_task_type: str,
                               query_data: dict,
                               plot_data: list, execute_date: str,
                               query_plot_analysis_file: str = None,
                               query_news_summary_file: str = None,
                               query_news_citation_file: str = None,
                               query_analysis_text_file: str = None):
    """
        作为文本分析chain的pipeline
        Args:
            strategy
            query_task: company/industry,需要区分个股还是行业分析的任务，给出任务目标名称

            query_data: 请求本任务的输入数据
            plot_data: 图表数据

            execute_date: 执行日期

            query_plot_analysis_file: 图表分析的结果和数据引用
            query_news_summary_file: summary结果的文件路径
            query_news_citation_file: 引用结果的文件路径
            query_analysis_text_file: 完整的analysis_text结果的文件路径

        Returns:
            result:  封装好的text结果
    """
    # step 1: company/industry任务的区分和判断
    if query_task_type == "company":
        assert query_task in task1_A_company
    elif query_task_type == "industry":
        assert query_task in task2_A_industry
    else:
        raise TypeError("company和industry都没有传合法的值")

    # step 2: 配置中间变量的存储路径
    query_plot_analysis_file = data_config['text_analysis_chain']['query_plot_analysis_file'].format(date=execute_date,
                                                                                                     query_task=query_task) \
        if query_plot_analysis_file is None else query_plot_analysis_file

    query_news_summary_file = data_config['text_analysis_chain']['query_news_summary_file'].format(date=execute_date,
                                                                                                   query_task=query_task) \
        if query_news_summary_file is None else query_news_summary_file
    query_news_citation_file = data_config['text_analysis_chain']['query_news_citation_file'].format(date=execute_date,
                                                                                                     query_task=query_task) \
        if query_news_citation_file is None else query_news_citation_file
    query_analysis_text_file = data_config['text_analysis_chain']['query_analysis_text_file'].format(date=execute_date,
                                                                                                     query_task=query_task) \
        if query_analysis_text_file is None else query_analysis_text_file

    # step 3: plot分析的chain
    path = Path(query_plot_analysis_file)
    if not path.exists():
        print("run plot analysis.")
        run_plot_analyse_chain(plot_data=plot_data, query_task_type=query_task_type,
                               output_file=query_plot_analysis_file)
    else:
        print("plot analysis is existent.")

    assert strategy is not None
    # step 4: news的summary chain
    if check_if_run(query_news_summary_file, strategy):
        print("run news summary.")
        run_news_summary_chain(
            strategy=strategy,
            query_task=query_task,
            query_task_type=query_task_type,
            query_data=query_data,
            output_file=query_news_summary_file
        )
    else:
        print("news summary is existent.")

    # step 5: news的citation chain
    if check_if_run(query_news_citation_file, strategy):
        print("run news citation.")
        run_citation_extraction_chain(
            strategy=strategy,
            query_data=query_data,
            query_news_summary_file=query_news_summary_file,
            output_file=query_news_citation_file
        )
    else:
        print("news citation is existent.")

    # step 6: analysis中citation的merge chain
    if check_if_run(query_analysis_text_file, strategy):
        print("run merge.")
        result = run_source_extraction_merge_plot_analysis(
            strategy=strategy,
            query_task_type=query_task_type,
            query_plot_analysis_file=query_plot_analysis_file,
            query_news_citation_file=query_news_citation_file,
            query_news_summary_file=query_news_summary_file,
            output_file=query_analysis_text_file
        )
        return result
    else:
        with open(query_analysis_text_file, "r", encoding='utf-8') as json_file:
            data_list = json.load(json_file)
        for data in data_list:
            if data["strategy"] == strategy:
                return data
    return None

def check_if_run(file_path:str, strategy:str):
    path = Path(file_path)
    if not path.exists():
        return True
    with open(file_path, "r", encoding='utf-8') as json_file:
        dict_list = json.load(json_file)
    for dict_data in dict_list:
        if dict_data["strategy"] == strategy:
            return False
    return True


if __name__ == "__main__":
    # origin_data = {'ids': [['id87', 'id119', 'id70']],
    #                'distances': None,
    #                'metadatas': [{
    #                    'source': 'eastmoney',
    #                    'title': '固态电池大火 14家券商扎堆推荐宁德时代 新能源车龙头ETF（159637）涨超2%',
    #                    'url': 'https://finance.eastmoney.com/a/202404093037417958.html'},
    #                    {
    #                        'source': 'eastmoney',
    #                        'title': '固态电池大火 14家券商扎堆推荐宁德时代 新能源车龙头ETF（159637）涨超2%',
    #                        'url': 'https://finance.eastmoney.com/a/202404093037417958.html'},
    #                    {
    #                        'source': 'eastmoney',
    #                        'title': '新能源重磅利好！外资突然唱多宁德时代 黄仁勋、奥特曼：AI的尽头是光伏和储能',
    #                        'url': 'https://finance.eastmoney.com/a/202403103007210340.html'}],
    #                'embeddings': None,
    #                'documents':
    #                    ['2024-04-24，据宁德时代微博消息，宁德时代将于4月25日召开2024宁德时代储能新品发布会。',
    #                     '2024-04-24，据宁德时代微博消息，宁德时代将于4月25日召开2024宁德时代储能新品发布会。',
    #                     '2024-03-25，【曾毓群：宁德时代与特斯拉在合作开发充电速度更快的电池】宁德时代董事长曾毓群在接受采访时表示，宁德时代与特斯拉在合作开发充电速度更快的电池。曾毓群透露，双方正共同研究新型电化学结构等电池技术，旨在加快充电速度。曾毓群另外表示，宁德时代正为特斯拉位于美国内华达州的工厂提供设备。'],
    #                'uris': None, 'data': None,
    #                'included': ['metadatas', 'documents', 'distances']
    #                }
    # plot_data = [("daily", pd.read_csv(
    #     "/home/zmy/AFAC/research_report/intermediate_data/plot_data//stock/20240628//宁德时代//宁德时代近期股价走势.csv"))]

    # path = Path("../summary_generation/0701/query_data.json")
    # if not path.exists():
    #     DATE = '20240628'
    #     start_year, start_month, start_day, present_year, present_month, present_day = get_start_end_date(DATE)
    #     vector_db = VectorDBConnector(collection_name="finnews-data")
    #     get_news_from_api(query_content="宁德时代", keywords="新能源、锂电池、特斯拉", start_year=start_year,
    #                       start_month=start_month, start_day=1,
    #                       end_year=present_year, end_month=present_month, end_day=present_day, save_json=False,
    #                       save_chroma=True, vector_db=vector_db)
    #     prompt = "生成{}在{}的投资评级分析".format("宁德时代", "20240701")
    #     filtered_news_ids_list = filter_news_from_query_data(vector_db=vector_db, prompt=prompt,
    #                                                          topk=3, execute_date="20240628")
    #     query_data = vector_db.collection.get(ids=filtered_news_ids_list, include=["documents", "metadatas"])
    #     print(query_data)
    #     with open("../summary_generation/0701/query_data.json","w",encoding='utf-8')as f:
    #         json.dump(query_data, f, ensure_ascii=False, indent=4)
    # else:
    #     with open("../summary_generation/0701/query_data.json","w",encoding='utf-8')as f:
    #         query_data = json.load(f)

    DATE = '20240629'
    start_year, start_month, start_day, present_year, present_month, present_day = get_start_end_date(DATE)
    vector_db = VectorDBConnector(collection_name="finnews-data")
    get_news_from_api(query_content="亿纬锂能", keywords="锂电池、动力电池、固态电池", start_year=start_year,
                      start_month=start_month, start_day=1,
                      end_year=present_year, end_month=present_month, end_day=present_day, save_json=False,
                      save_chroma=True, vector_db=vector_db)
    prompt = "生成{}在{}的投资评级分析".format("亿纬锂能", "20240629")
    filtered_news_ids_list = filter_news_from_query_data(vector_db=vector_db, prompt=prompt,
                                                         topk=3, execute_date="20240629")
    query_data = vector_db.collection.get(ids=filtered_news_ids_list, include=["documents", "metadatas"])
    print(query_data)
    run_analysis_text_pipeline(strategy="summary10",
                               query_task="亿纬锂能",
                               query_task_type="company",
                               query_data=query_data,
                               plot_data=[],
                               execute_date="20240629")
                               # query_plot_analysis_file="../summary_generation/0701/plot_analysis.json",
                               # query_news_summary_file="../summary_generation/0701/news_summary.json",
                               # query_news_citation_file="../summary_generation/0701/news_citation.json",
                               # query_analysis_text_file="../summary_generation/0701/text_analysis.json"
                               # )
