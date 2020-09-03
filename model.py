#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 * @Author : felixfeng
 * @Createdate : 2020/4/7  1:53 PM
 * @Program : chinese_newword_discovery
 * @Description : 
"""
import re
import numpy as np
from collections import defaultdict
import string
from tqdm import tqdm
import math


def is_good(w):
    '''
    用规则来干掉一些统计学较难处理的词。
    :param w:
    :return:
    '''
    if isChinese(w) \
            and not re.findall(u'[较很越增]|[多少大小长短高低好差]', w) \
            and not u'的' in w \
            and not u'了' in w \
            and not u'这' in w \
            and not u'那' in w \
            and not u'到' in w \
            and not w[-1] in u'为一人给内中后省市局院上所在有与及厂稿下厅部商者从奖出' \
            and not w[0] in u'每各该个被其从与及当为' \
            and not w[-2:] in [u'问题', u'市场', u'邮件', u'合约', u'假设', u'编号', u'预算', u'施加', u'战略', u'状况', u'工作', u'考核',
                               u'评估', u'需求', u'沟通', u'阶段', u'账号', u'意识', u'价值', u'事故', u'竞争', u'交易', u'趋势', u'主任',
                               u'价格', u'门户', u'治区', u'培养', u'职责', u'社会', u'主义', u'办法', u'干部', u'员会', u'商务', u'发展',
                               u'原因', u'情况', u'国家', u'园区', u'伙伴', u'对手', u'目标', u'委员', u'人员', u'如下', u'况下', u'见图',
                               u'全国', u'创新', u'共享', u'资讯', u'队伍', u'农村', u'贡献', u'争力', u'地区', u'客户', u'领域', u'查询',
                               u'应用', u'可以', u'运营', u'成员', u'书记', u'附近', u'结果', u'经理', u'学位', u'经营', u'思想', u'监管',
                               u'能力', u'责任', u'意见', u'精神', u'讲话', u'营销', u'业务', u'总裁', u'见表', u'电力', u'主编', u'作者',
                               u'专辑', u'学报', u'创建', u'支持', u'资助', u'规划', u'计划', u'资金', u'代表', u'部门', u'版社', u'表明',
                               u'证明', u'专家', u'教授', u'教师', u'基金', u'如图', u'位于', u'从事', u'公司', u'企业', u'专业', u'思路',
                               u'集团', u'建设', u'管理', u'水平', u'领导', u'体系', u'政务', u'单位', u'部分', u'董事', u'院士', u'经济',
                               u'意义', u'内部', u'项目', u'建设', u'服务', u'总部', u'管理', u'讨论', u'改进', u'文献', u'师傅'] \
            and not w[:2] in [u'考虑', u'图中', u'每个', u'出席', u'一个', u'随着', u'不会', u'本次', u'产生', u'查询', u'是否', u'作者'] \
            and not (u'博士' in w or u'硕士' in w or u'研究生' in w) \
            and not (len(set(w)) == 1 and len(w) > 1) \
            and not (w[0] in u'一二三四五六七八九十' and len(w) == 2) \
            and re.findall(u'[^一七厂月二夕气产兰丫田洲户尹尸甲乙日卜几口工旧门目曰石闷匕勺]', w) \
            and not u'进一步' in w:
        return True
    else:
        return False


def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            continue
        else:
            return False

    return True


def split_sentence(text):
    '''
    作用是分句
    :param text: content
    :return: list， 分好的句子
    '''
    punctuation_set = set(['?', '!', ';', '，', ',', '？', '！', '。', '；', '……', '…', '\n', ' '])
    sentence_set = []
    inx_position = 0  # 索引标点符号的位置
    char_position = 0  # 移动字符指针位置
    for char in text:
        char_position += 1
        if char in punctuation_set:
            next_char = list(text[inx_position:char_position + 1]).pop()
            if next_char not in punctuation_set:
                # 正则去掉奇怪的表情符号等
                sentence_set.append(re.sub('[^\u4e00-\u9fa50-9a-zA-Z]+',"",text[inx_position:char_position]))
                inx_position = char_position
    if inx_position < len(text):
        sentence_set.append(re.sub('[^\u4e00-\u9fa50-9a-zA-Z]+',"",text[inx_position:]))

    return sentence_set


def get_data(path):
    '''
    :param path: 语料的路径
    :return: 分好的句子
    '''
    texts = []
    with open(path,"r",encoding='utf-8') as f:
        lines = f.readlines()
        [texts.extend(split_sentence(line.strip())) for line in lines]
    return texts


def cut_step_1(texts, min_count, n):
    '''
    :param texts: 分好句的内容
    :param min_count:   判断词汇最少频次
    :param n:   内聚计算力度
    :return:    词汇及词汇统计数
    '''
    ngrams = defaultdict(int)
    for t in texts:
        for i in range(len(t)):
            for j in range(1, n+1):
                if i+j <= len(t):
                    ngrams[t[i:i+j]] += 1

    ngrams = {i:j for i,j in ngrams.items() if j >= min_count}
    total = 1.*sum([j for i,j in ngrams.items() if len(i) == 1])
    return ngrams,total


def cohesion_step(ngrams,total,min_num=2):
    '''
    :param ngrams: 词汇及词汇统计数
    :param total: 单字词的统计数
    :return: 通过内聚审核的词汇及词汇数
    '''
    min_proba = {2:5, 3:25, 4:125}

    def cohesion_score(s):
        '''
        :param s:
        :param min_proba: 不同字长的分数限制
        :return:
        '''
        if len(s) >= min_num:
            score = min([total*ngrams[s]/(ngrams[s[:i+1]]*ngrams[s[i+1:]]) for i in range(len(s)-1)])
            if score > min_proba[len(s)]:
                return True
        else:
            return False

    ngrams_ = set(i for i,j in ngrams.items() if cohesion_score(i))
    return ngrams_


def cut_step_2(texts, ngrams, total, min_count, n, min_num):
    '''
    :param texts: 粗分
    :param ngrams_: 内聚计算后，符合条件的词汇
    :param min_count:
    :param n:
    :return:
    '''
    words = defaultdict(int)
    # 新版 pmi计算公式及阈值判断 min_proba = {2:5, 3:25, 4:125}
    ngrams_ = cohesion_step(ngrams,total,min_num)

    def cut(s):
        r = np.array([0]*(len(s)-1))
        for i in range(len(s)-1):
            for j in range(2, n+1):
                if s[i:i+j] in ngrams_:
                    r[i:i+j-1] += 1
        w = [s[0]]
        for i in range(1, len(s)):
            if r[i-1] > 0:
                w[-1] += s[i]
            else:
                w.append(s[i])
        return w

    def is_real(s):
        if len(s) >= 3:
            for i in range(3, n + 1):
                for j in range(len(s) - i + 1):
                    if s[j:j + i] not in ngrams_:
                        return False
            return True

    for t in texts:
        if not t:
            continue
        for i in cut(t):
            words[i] += 1

    words = {i:j for i,j in words.items() if j >= min_count and is_real(i) and is_good(i)}

    return words


def new_words_extract(path,min_count=30, n=4, min_num = 2, min_entropy=0.8):
    '''
    :param path: 存放data的文件路径
    :param min_count: 新词的最小重复次数
    :param n: 内聚力计算长度
    :param min_num: 最小成词字数长度为2及2以上个字，但最终会是3个字，这里的2是用来将句子分隔的更准确，从而找到4个字以上的词汇
    :param min_entropy: 左右熵的最小阈值
    :return: 回一个dict，其中包含新词及词频
    感谢以下项目及作者:
    北交大 Gaohui Shang: 《Research on Chinese New Word Discovery Algorithm Based on Mutual Information》
    苏剑林老师： https://github.com/bojone/word-discovery
    '''
    #目前没有合适mi大于4的超参，同时4字算内聚已足够，仍旧可以找出大于4个字的词
    if n > 4: n = 4
    # 获取按句子分隔后的数据
    texts = get_data(path)
    # 首次切分，内聚力=4，最小出现词数=30
    ngrams,total = cut_step_1(texts, min_count, n)
    # 二次切分通过新版mi计算公式，新词成词字数大于3，及不同字数内聚力不同 min_proba = {2:5, 3:25, 4:125}
    words = cut_step_2(texts, ngrams, total, min_count, n, min_num)
    # 左右熵判断
    words_ = r_l_entropy(texts,words,min_entropy)
    # 返回一个map，其中包含新词及词频
    return words_


def words_rl_count_ini(words):
    '''
    :param words: 需要被计算左右熵的词汇
    :return: 初始化的记录map
    '''
    words_rl_count = {}
    for word in words:
        if word not in words_rl_count:
            words_rl_count[word] = [defaultdict(int),defaultdict(int)]

    return words_rl_count


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def r_l_entropy(texts,words,min_entropy):
    '''
    :param min_entropy: 左右熵最小阈值
    :param texts: 原始文本
    :param words: 候选词
    :return: 候选词中左右熵最小值大于 min_entropy 的
    '''
    flashtext = MyFlashtext()
    flashtext.add_keywords_from_list(words)
    words_rl_count = words_rl_count_ini(words)

    def add2count(result,text):
        length = len(text)
        for item in result:
            word = item[0]
            l = item[1]-1
            r = item[2]
            if l >= 0:
                l_word = text[l]
                words_rl_count[word][0][l_word] += 1
            if r < length:
                r_word = text[r]
                words_rl_count[word][1][r_word] += 1

    for text in tqdm(texts):
        result = flashtext.extract_keywords(text,span_info=True)
        add2count(result,text)

    def entropy(f):
        # 计算左右熵
        t = sum([i for i in f.values()])
        ent=(-1)*sum([i/t*math.log(i/t) for i in f.values()])
        return ent

    for word,lr in words_rl_count.items():
        left_entropy=entropy(lr[0])
        right_entropy=entropy(lr[1])
        score = min(left_entropy,right_entropy)
        if score < min_entropy:
            words.pop(word)

    return words


class MyFlashtext(object):
    '''
    方法来源于 https://github.com/vi3k6i5/flashtext
    存在问题是对中文match出现匹配异常的bug，重写extract_keywords 方法后已修正。
    '''
    def __init__(self, case_sensitive=False):
        self._keyword = '_keyword_'
        self._white_space_chars = set(['.', '\t', '\n', '\a', ' ', ','])
        self.keyword_trie_dict = dict()
        self.case_sensitive = case_sensitive
        self._terms_in_trie = 0
        self.non_word_boundaries = set(string.digits + string.ascii_letters + '_')

    def __getitem__(self):
        return self.keyword_trie_dict

    def __setitem__(self, keyword, clean_name=None):
        status = False
        if not clean_name and keyword:
            clean_name = keyword

        if keyword and clean_name:
            if not self.case_sensitive:
                keyword = keyword.lower()
            current_dict = self.keyword_trie_dict
            for letter in keyword:
                current_dict = current_dict.setdefault(letter, {})
            if self._keyword not in current_dict:
                status = True
                self._terms_in_trie += 1
            current_dict[self._keyword] = clean_name
        return status

    def add_keyword(self, keyword, clean_name=None):
        return self.__setitem__(keyword, clean_name)

    def get_dict(self):
        return self.__getitem__()

    def add_keywords_from_list(self, keyword_list):
        # if not isinstance(keyword_list, list):
        #     raise AttributeError("keyword_list should be a list")

        for keyword in keyword_list:
            self.add_keyword(keyword)

    def extract_keywords(self, sentence, span_info=False):
        keywords_extracted = []
        if not sentence:
            # if sentence is empty or none just return empty list
            return keywords_extracted
        if not self.case_sensitive:
            sentence = sentence.lower()
        current_dict = self.keyword_trie_dict
        sequence_start_pos = 0
        sequence_end_pos = 0
        idx = 0
        sentence_len = len(sentence)
        while idx < sentence_len:
            char = sentence[idx]
            if self._keyword in current_dict or char in current_dict:
                # update longest sequence found
                sequence_found = None
                longest_sequence_found = None
                is_longer_seq_found = False
                if self._keyword in current_dict:
                    sequence_found = current_dict[self._keyword]
                    longest_sequence_found = current_dict[self._keyword]
                    sequence_end_pos = idx
                # re look for longest_sequence from this position
                if char in current_dict:
                    current_dict_continued = current_dict[char]
                    idy = idx + 1
                    while idy < sentence_len:
                        inner_char = sentence[idy]
                        if self._keyword in current_dict_continued:
                            longest_sequence_found = current_dict_continued[self._keyword]
                            sequence_end_pos = idy
                            is_longer_seq_found = True
                        if inner_char in current_dict_continued:
                            current_dict_continued = current_dict_continued[inner_char]
                        else:
                            break
                        idy += 1
                    else:
                        # end of sentence reached.
                        if self._keyword in current_dict_continued:
                            # update longest sequence found
                            longest_sequence_found = current_dict_continued[self._keyword]
                            sequence_end_pos = idy
                            is_longer_seq_found = True
                    if is_longer_seq_found:
                        idx = sequence_end_pos
                    else:
                        idx += 1
                current_dict = self.keyword_trie_dict
                if longest_sequence_found:
                    keywords_extracted.append((longest_sequence_found, sequence_start_pos, idx))
            else:
                current_dict = self.keyword_trie_dict
                idx += 1
            if idx + 1 >= sentence_len:
                if self._keyword in current_dict:
                    sequence_found = current_dict[self._keyword]
                    keywords_extracted.append((sequence_found, sequence_start_pos, sentence_len))
            sequence_start_pos = idx
        if span_info:
            return keywords_extracted
        return [value[0] for value in keywords_extracted]


if __name__=='__main__':
    # 文件路径
    path = ""
    # 返回格式dict {'你好':123} 词：出现次数, 这里最好再跟你的词库进行一个对比；或者把原文本基础分词后，找到那些不在基础分词中的词汇。
    words = new_words_extract(path)

    '''
    all_sen = get_data(path)
    all_word = set()
    # 这里使用的是jieba的全分词模式
    [[all_word.add(x) for x in jieba.cut(sen,cut_all=True)] for sen in all_sen] 
    w = [i for i, j in words.items() if i not in all_word]
    print(w)
    '''
