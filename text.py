# 文本分析实操
# 参考https://www.51cto.com/article/700850.html
import pandas as pd
import jieba
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import sys
import run
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import collections
import bar_chart_race as bcr
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import ward, dendrogram, linkage


# 主函数
# @run.change_dir
def main():
    drawWordCloud(endyear = 2003)
    combinePics()
    # generateCipin(year = 1958)
    # cipins = makeCipin(endyear = 1960)
    # bcr.bar_chart_race(cipins, steps_per_period=30, period_length=1500, title="人民日报年度词频可视化", bar_size=0.8, fixed_max=10, n_bars=10)
    # clusteText(endyear = 2003)
        
        
# 生成图云
def drawWordCloud(fromyear = 1957, endyear = 2003):
    for year in range(fromyear, endyear):
        # 加载词频结果
        cipin = generateCipin(year)
        # 生成词云
        wordcloud = WordCloud(font_path="./simhei.ttf", collocations = False, max_words = 100, min_font_size=10, max_font_size=500, background_color="white")
        wordcloud.generate_from_frequencies(cipin)
        wordcloud.to_file("./output/" + str(year) + '词云图.png')
        
       
# 合并图片
def combinePics(path = "./output"):
    # 生成文件列表
    filelist = []
    for year in range(1957, 2003):
        files = generateFileList(year = year, path = path)
        for file in files:
            filelist.append(file)
    # print(filelist)
    # 合并词云图
    """
    n = len(filelist)
    toImage = Image.new('RGBA',(500*12,300*12))
    for i in range(n):
        fromImge = Image.open(filelist[i])
        loc = (int(i%4)*500, (i/4)*300)
        print(loc, filelist[i])
        toImage.paste(fromImge, loc)
        
    toImage.save("merged.png")
    """
    img = ''
    for i, file in enumerate(filelist):
        if i == 0:
            img = Image.open(file)
            img_array = np.array(img)
        else:
            img_array2 = np.array(Image.open(file))
            img_array = np.concatenate((img_array, img_array2), axis = 0)
            img = Image.fromarray(img_array)
            
    img.save("merged.png")
        
    
    
# 创建停用词列表
def stopWordsList():
    wordlist = "./stopwords.txt"
    stopwords = [line.strip() for line in open(wordlist, 'r', encoding='utf-8').readlines()]
    return stopwords[0].strip()
    
   
# 生成文件列表
def generateFileList(year, path = "./paper"):
    filelist = []
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for one in files:
                if one.find(str(year)) >= 0:
                    filelist.append(os.path.join(root, one))
        # print(len(filelist))
    else:
        print("无效目录")
    
    return filelist
    
    
# 加载文本，进行分词
def cutText(year):
    print("正在处理", year, "年内容。加载文本内容，进行分词")
    filelist = generateFileList(year)
    n = len(filelist)
    if n == 0:
        return None

    stopwords = stopWordsList()
    outstr = []
    for i in range(n):
        print(year, "年数据加载进度:", float(i/n)*100, "%")
        path = filelist[i]
        with open(path, "r", encoding = "UTF-8") as file:
            data = file.read()
        # print(data)
        str = re.sub('[^\w]', '', data)
        # print(str)
        text_cut = jieba.lcut(str)
        # print(text_cut)
    
        for word in text_cut:
            if word not in stopwords:
                outstr.append(word)
        # print("处理后分词结果", outstr)
    
    # 保存分词结果到文件
    saveText(year, outstr)
    del(outstr)
    return
    
    
# 将列表保存为文本文件
# @run.change_dir
def saveText(year, text):
    path = "./cutwords/" + str(year) + ".txt"
    with open(path, "w") as file:
        for item in text:
            file.write(item)
            file.write(" ")
            
            
# 从文件中加载分词结果
# @run.change_dir
def loadText(year):
    path = "./cutwords/" + str(year) + ".txt"
    with open(path, "r", encoding = "UTF-8") as file:
        data = file.read()
    outstr = data.strip(" ")
    return outstr
    
    
# 生成分词数据
def generateData(fromyear = 1957, endyear = 2003):
    for year in range(fromyear, endyear):
        cutText(year)
        # print("变量大小", sys.getsizeof(outstr)/(1024)**2, "MB")
        
        
# 统计词频
def generateCipin(year):
    text = loadText(year)
    text = text.split(" ")
    word_counts = collections.Counter(text)
    return word_counts
    
    
# 生成词频数据集合
def makeCipin(fromyear = 1957, endyear = 2003):
    cipins = []
    years = []
    results = pd.DataFrame()
    for year in range(fromyear, endyear):
        years.append(year)
        cipin = generateCipin(year)
        cipins.append(cipin)
        # print(year, cipin.keys, cipin.values, dict(cipin.most_common(100)))
        results = results.append(dict(cipin.most_common(100)), ignore_index = True)
        # print(results.head())
        # input("按任意键继续")
    #results = pd.DataFrame({"年度":years, "词频":cipins})
    results["年度"] = years
    results.set_index("年度", inplace = True)
    print(results.info(), results.head())
    return results
    
    
# 文本聚类
def clusteText(fromyear = 1957, endyear = 2003):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, max_features=3000, max_df=0.99, min_df=0.1)
    raw_str = []
    for year in range(fromyear, endyear):
        text = loadText(year)
        raw_str.append(text)
    # raw_str = raw_str.split(" ")
    print(len(raw_str))
    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_str)
    print(year, tfidf_matrix.shape)
    # 计算文档相似性
    dist = 1 - cosine_similarity(tfidf_matrix)
    print("文档相似性", dist)
    # 获得分类
    linkage_matrix = linkage(dist, method='ward', metric='euclidean', optimal_ordering = False)
    print(linkage_matrix)
    # 可视化
    plt.figure(figsize = (25, 10))
    plt.title("人民日报全文聚类")
    dendrogram(
        linkage_matrix,
        labels = [str(year) for year in range(fromyear, endyear)],
        leaf_rotation=-70,
        leaf_font_size=12
    )
    plt.savefig("./output/cluste.jpg")
    plt.close()
    

if __name__ == "__main__":
    # generateData(endyear = 1959)
    main()
    # filelist = generateFileList(1957)
    