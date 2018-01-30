# -*- coding: utf-8 -*-
from __future__ import division
import pickle
import requests

import eigen_config
from simplex import utils, logger
from simplex.model import KeyWordClassifier
from simplex.utils.finance_util import parse_report

class FinancePreprocessV1(object):
    def __init__(self, app_config):
        config = app_config
        self.finance_parse_host = config.get("finance_parse_host")
        self.finance_du_host = config.get("finance_du_host")
        self.version = config.get("version")
        self.risk_words = ['风险提示','风险']

    def _get_features_from_du_host(self, docs):
        try:
            ret = requests.post(self.finance_du_host, json=docs, timeout=30)
        except requests.ReadTimeout:
            logger.warning("Time out when try to get finance docs features")
            return []
        if ret.status_code == 200:
            return ret.json()

    def get_features(self, docs):
        batch_size = 20
        batch = []
        results = []
        for doc in docs:
            batch.append(doc)
            if len(batch) == batch_size:
                ret = self._get_features_from_du_host(batch)
                if ret:
                    results.extend(ret)
                batch = []
        if len(batch) != 0:
            ret = self._get_features_from_du_host(batch)
            if ret:
                results.extend(ret)
        return results

    def parse_pdf(self, url):
        try:
            payload = {'url':url}
            ret = requests.get(self.finance_parse_host, params=payload, timeout=300)
        except:
            return []
        
        if ret.status_code == 200:
            return ret.json()

    def is_risk_paragraph(self, headline):
        headline = ' '.join(headline)
        for word in self.risk_words:
            if word in headline:
                return True
        return False

    def feature_process(self, item):
        '''处理财报pdf，解析为段落，标注意图
        Args:
            item: 原始财报pdf内容，应该包括
                - url: pdf的url
                - articleid
                - name
                - code
                - year
                - quarter
                - 其他
        '''
        url = item['url']
        paragraphs = self.parse_pdf(url)

        if not paragraphs:
            return None

        docs_raw = [{'content':p['content']} for p in paragraphs]
        headlines_raw = [{'content':p['headline'][-1] if p['headline'] else ''} for p in paragraphs]

        docs = self.get_features(docs_raw)
        headlines = self.get_features(headlines_raw)

        for i,hd in enumerate(headlines):
            docs[i]['title'] = ' ## '.join(paragraphs[i]['headline'])
            docs[i]['seq'] = paragraphs[i]['seq']
            docs[i]['report_type'] = 'financial'
            risk = self.is_risk_paragraph(paragraphs[i]['headline'])

            for j,intent in enumerate(hd['features']['intents']):
                if not risk:
                    docs[i]['features']['intents'][j]['prob'] += intent['prob']
                else:
                    docs[i]['features']['intents'][j]['prob'] = 0.0

            # risk paragraphs
            if risk:
                 docs[i]['features']['intents'][15]['prob'] = 1.0

            docs[i]['article'] = item['articleid']
            docs[i]['id'] = '{0}_{1}'.format(item['articleid'],i)

        # remove paragraph that is belong to '其他' intent
        docs = [doc for doc in docs if doc['features']['intents'][14]['prob'] < 1.0]

        # append other attribute back to doc
        keys = [key for key in item.keys() if key not in [
            'articleid', 'content']]
        for doc in docs:
            doc.update({k: item[k] for k in keys})
            # add version
            doc.update({"model_version":self.version})
        return docs

class FinancePreprocess(object):
    def __init__(self, app_config):
        config = app_config
        self.finance_du_host = config.get("finance_du_host")
        self.version = config.get("version")

        self.stock_types = ['hushen', 'xinsanban']

        # load stock info data
        stock_info_oss = config.get("stock_info_oss")
        stock_info_local = utils.oss_to_local(stock_info_oss, "/tmp")
        stock_info = pickle.load(open(stock_info_local, 'rb'))
        self.stock_name2id = stock_info['n2i']
        self.stock_name2type = stock_info['n2t']
        self.stock_kw_classifier = KeyWordClassifier(
            weighted=False, keywords=self.stock_name2id.keys())

        key_words = ['发布', '营业收入', '营收', '归属于上市公司股东的净利润',
                     '营业利润', '归属母公司净利润', '净利润', '归母净利润', '同比', '财务报告']
        quarter_keys = [
            ['q1', '第一季度', '一季度', '一季报'],
            ['q2', '半年度', '半年报'],
            ['q3', '第三季度', '三季度', '三季报'],
            ['q4', '年度报告', '年报']
        ]

        self.content_kw_classifier = KeyWordClassifier(
            weighted=False, keywords=key_words)
        self.quarter_kw_classifiers = [KeyWordClassifier(
            weighted=False, keywords=kws) for kws in quarter_keys]

    def _get_features_from_du_host(self, docs):
        try:
            ret = requests.post(self.finance_du_host, json=docs, timeout=10)
        except requests.ReadTimeout:
            logger.warning("Time out when try to get finance docs features")
            return []
        if ret.status_code == 200:
            return ret.json()

    def get_features(self, docs):
        batch_size = 20
        batch = []
        results = []
        for doc in docs:
            batch.append(doc)
            if len(batch) == batch_size:
                ret = self._get_features_from_du_host(batch)
                if ret:
                    results.extend(ret)
                batch = []
        if len(batch) != 0:
            ret = self._get_features_from_du_host(batch)
            if ret:
                results.extend(ret)
        return results

    def feature_process(self, item):
        '''处理原始的文章，将其划分为段落，并进行意图分类

        Args:
            item: 包含原始文章的所有信息的dict,要求必须包括的内容有：
                - articleid: 文章id
                - content: 文本内容
                - pubdate: 发布日期
                - title: 文章标题
                - source: 文章来源，要求为以下四种：
                    - hushen: A股
                    - xinsanban: 新三板
                    - jiemodui*: 芥末堆研报
                    - tonghuashun**: 同花顺研报
                    - 未来其他的源
                - name: 股票名称
                - code: 股票代码
                - year: 年份
                - quarter: 季度
            *如果文章类型为jiemodui，不需要包括股票名称在内之后的信息，会自动解析判断
            **如果文章类型为tonghuashun，不需要年份、季度信息，会自动解析判断

        Return:
            如果输入是财报或者能够解析出年份季度的研报，将会返回docs。否则返回None，无需处理。
            docs: 分段后的结果，是一个list of doc，其中每个doc包含以下这些key:
                - id: 段落唯一标识
                - content: 段落内容
                - title: 段落标题
                - pubdate: 发布日期
                - seq: 段落在文章中的位置
                - source: 文章来源，同输入
                - report_type: 文章类型，包括以下两种：
                    - financial: 公司发布的财报
                    - research: 研报，如同花顺上的研报，芥末堆的研报
                - name: 股票名称
                - code: 股票代码
                - year: 财报年份
                - quarter: 财报季度
                - model_version: 模型版本号
                - article: 文章id
                - features: 段落特征，只需要插入到SQL表格中，无需插入ES
        '''
        articleid = item['articleid']
        source_type = item['source']
        content = item['content']
        pubdate = item['pubdate']
        title = item['title']

        stock_name = item.get('name', None)
        stock_id = item.get('code', None)
        year = int(item.get('year', 0))
        quarter = int(item.get('quarter', 0))

        # split the content into paragraphs
        if source_type in self.stock_types:
            # do nothing here
            pass
        elif source_type == 'tonghuashun':
            ret = self.get_stock_info_from_research(
                stock_name, stock_id, title, content, pubdate)
            if ret:
                stock_id, stock_name, year, quarter = ret
            else:
                return None

        elif source_type == 'jiemodui':
            ret = self.get_stock_info_from_jmd(title, content, pubdate)
            if ret:
                stock_id, stock_name, year, quarter = ret
            else:
                return None
        else:
            raise ValueError(
                "source type of {0} is not supported yet.".format(source_type))

        logger.info(u"start to parse article {0} with source type {1}, stock name {2}, year {3}, quarter {4}, version {5}".format(
            articleid, source_type, stock_name, year, quarter, self.version))
        docs = parse_report(articleid, content, source_type,
                            stock_name, stock_id, year, quarter, pubdate, self.version)
        if docs:
            docs = self.get_features(docs)
            return docs
        else:
            return None

    def predict_year_quarter(self, pubdate):
        '''根据文章发布日期预测大致的年份和季度
        '''
        # in case pubdate is YYYY-MM-DDTHH:MM:SS
        pubdate = pubdate.split("T")[0]
        year, month, day = map(int, pubdate.split("-"))

        # 一般公司的报告会在季度之后才会发布，因此预测季度应该为当前发布日期季度减1
        quarter = (month - 1) // 3
        # 上一年年报
        if quarter == 0:
            quarter = 4
            year -= 1
        return year, quarter

    def predict_from_content(self, content, pubdate):
        '''根据文章内容进一步判断年份和季度
        '''
        # get the predicted year and quarter
        year, quarter = self.predict_year_quarter(pubdate)

        # only use the first two paragraphs to predict
        content = "".join(content.split("\n")[:2])

        predict, _ = self.content_kw_classifier.predict(content, method=1)
        # no key words found in the content
        if predict < 1.0:
            return None

        # if we found quarter key words in the content, return the year and quarter
        for i in range(4):
            predict_quarter, _ = self.quarter_kw_classifiers[i].predict(
                content, method=1)
            if predict_quarter >= 1.0:
                # predicted quarter is later than pubdate quarter
                # this should be last year's report
                if quarter < i + 1:
                    return year - 1, i + 1
                return year, i + 1

        # if we found at least 2 key words in the content, return the predicted year and quarter
        # else we can not tell if this is really a article about finance report
        if predict >= 2.0:
            return year, quarter
        else:
            return None

    # pubdate shoule be in format YYYY-MM-DD
    def get_stock_info_from_jmd(self, title, content, pubdate):
        """解析芥末堆数据
        """
        title = title.replace(" ", "")
        # 只采用财报季文章
        if u"【财报季】" not in title:
            return None

        _, stocks = self.stock_kw_classifier.predict(title, method=1)
        # 不处理没有股票，或者存在多个股票的情况
        if len(stocks) != 1:
            return None

        ret = self.predict_from_content(content, pubdate)

        if ret:
            year = ret[0]
            quarter = ret[1]
            stock_name = stocks[0]
            stock_id = self.stock_name2id[stock_name]
            return stock_id, stock_name, year, quarter
        return None

    def get_stock_info_from_research(self, stock_name, stock_id, title, content, pubdate):
        '''解析研报获得股票年份信息
        '''
        ret = self.predict_from_content(content, pubdate)
        if ret:
            year = ret[0]
            quarter = ret[1]
            return stock_id, stock_name, year, quarter
        return None
