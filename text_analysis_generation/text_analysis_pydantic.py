from pydantic import BaseModel, Field


class CompanySummary(BaseModel):
    总结: str = Field(descrpition="简化后的投资评级分析", min_length=179, max_length=229)

class IndustrySummary(BaseModel):
    总结: str = Field(descrpition="简化后的投资评级分析", min_length=231, max_length=381)

class PlotAnalyse(BaseModel):
    """分析图表数据并给出一句话总结分析"""
    分析: str = Field(description="对图表列表的总结分析", max_length=50)

class Citation(BaseModel):
    """寻找文本中的引用内容."""
    总结文本片段: str = Field(description="总结文本中的内容")
    来源: str = Field(description="参考来源的标题")

class CitationList(BaseModel):
    """抽取引用信息列表."""
    citation_list: list[Citation] = Field(description="文本引用标注信息的列表")


class SourceInfo(BaseModel):
    """抽取新闻来源"""
    来源: str = Field(description="新闻来源")
