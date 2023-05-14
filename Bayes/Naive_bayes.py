from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def bayes_news():
    """
    朴素贝叶斯对新闻分类
    :return:
    """
    # 1.获取数据
    news = fetch_20newsgroups(subset='all')
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)
    print(news.target)
    # 3.特征工程 - 文本特征抽取Tf-idf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.朴素贝叶斯算法预估流程
    estimator = MultinomialNB(alpha=1.0)
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 方法1: 直接和真实值比对
    y_predict= estimator.predict(x_test)
    print('y_predict:', y_predict)
    print('直接比对真实值和测试值:\n', y_test == y_predict)
    # 方法2: 计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率: ', score)

    return None

if __name__ == "__main__":
    bayes_news()