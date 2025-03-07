import re
import json


"""计算指标的类，初始化无需带参数"""
class Calculate_rate:
    """读取文件内容"""
    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    """逐句获取input_txt的role: content，以列表形式输出"""
    def preprocess_json_to_list_with_role(self, input_dir):
        # 读取 original_result.txt 文件的内容
        with open(input_dir, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 遍历 sentence_list 并格式化输出
        result = []
        for sentence in data['sentence_list']:
            formatted_sentence = f"{sentence['role']}: {sentence['content']}"
            result.append(formatted_sentence)
        return result

    """将txt格式存储的role: content内容中的全部content单独取出来生成完整的字符串"""
    def preprocess_text(self, text):
        """移除无关字符"""
        # 移除行首的"1:"、"2:"等标识符
        text = re.sub(r'^\d+:\s*', '', text, flags=re.MULTILINE)
        # 移除换行符
        text = text.replace('\n', '')
        return text

    """计算文本中出现的热词数量"""
    def count_hotwords(self, text, hotwords):
        count = 0
        for hotword in hotwords:
            # 使用正则表达式搜索热词出现的次数
            count += len(re.findall(re.escape(hotword), text))
            if hotword == "DR":
                count += len(re.findall(re.escape("Dr"), text))
                count += len(re.findall(re.escape("dr"), text))
        return count

    """计算文本中出现的各个热词的频次"""
    def count_hotword_rate(self, text, hotwords):
        count = {}
        for hotword in hotwords:
            # 使用正则表达式搜索热词出现的次数
            count[hotword] = len(re.findall(re.escape(hotword), text))
            if hotword == "DR":
                count[hotword] += len(re.findall(re.escape("Dr"), text))
                count[hotword] += len(re.findall(re.escape("dr"), text))
        return count

    """对指定文件进行热词频次计算"""
    def calc_hotword_rate(self, check_file, hotwords_dir):
        hotwords = []
        # 打开文件并读取每一行
        with open(hotwords_dir, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行末的换行符
                cleaned_line = line.strip()
                hotwords.append(cleaned_line)

        check_text = self.read_file(check_file)
        check_text = self.preprocess_text(check_text)
        # 计算热词频次
        hotwords_rate = self.count_hotword_rate(check_text, hotwords)
        return hotwords_rate

    """对指定文件计算新增热词替换数"""
    def calculate_replacement_rate(self, original_file, result_file, hotwords_dir):
        hotwords = []
        # 打开文件并读取每一行
        with open(hotwords_dir, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行末的换行符
                cleaned_line = line.strip()
                hotwords.append(cleaned_line)

        original_text_list = self.preprocess_json_to_list_with_role(original_file)
        original_text = ""
        for text in original_text_list:
            original_text += text

        result_text = self.read_file(result_file)

        original_text = self.preprocess_text(original_text)
        result_text = self.preprocess_text(result_text)
        origin_hotword_count = self.count_hotwords(original_text, hotwords)
        result_hotword_count = self.count_hotwords(result_text, hotwords)
        # 计算替换率
        replacement_rate = result_hotword_count - origin_hotword_count
        return replacement_rate, origin_hotword_count, result_hotword_count

