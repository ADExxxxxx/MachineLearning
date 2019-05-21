import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

# K-Means模型
iris = load_iris()
iris_data = iris['data']
iris_target = iris['target']
iris_names = iris['feature_names']


scale = MinMaxScaler().fit(iris_data)
iris_dataScale = scale.transform(iris_data)

kmeans = KMeans(n_clusters=3, random_state=123).fit(iris_dataScale)

tsne = TSNE(n_components=2, init='random', random_state=177).fit(iris_data)
df = pd.DataFrame(tsne.embedding_)
df['labels'] = kmeans.labels_
df1 = df[df['labels'] == 0]
df2 = df[df['labels'] == 1]
df3 = df[df['labels'] == 2]

fig = plt.figure(figsize=(9, 6))
plt.plot(df1[0], df1[1], "bo", df2[0],
         df2[1], "r*", df3[0], df3[1], "gD")
plt.savefig('image/聚类分布.png')
plt.show()

result = []
for i in range(2, 7):
    kmeans = KMeans(n_clusters=i,random_state=123).fit(iris_data)
    score = fowlkes_mallows_score(iris_target,kmeans.labels_)
    temp = "iris数据聚" + str(i) + "类FMI评价分值为: " + str(score) + "\n"
    result.append(temp)

with open("data/K-Means模型FMI评价.txt", "w") as f:
    for i in range(len(result)):
        f.writelines(result[i])

silhouetteScore = []
for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, random_state=123).fit(iris_data)
    score = silhouette_score(iris_data, kmeans.labels_)
    silhouetteScore.append(score)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 15), silhouetteScore, linewidth=1.5, linestyle="-")
plt.savefig('image/轮廓系数评价.png')
plt.show()

result.clear()
for i in range(2, 7):
    kmeans = KMeans(n_clusters=i, random_state=123).fit(iris_data)
    score = calinski_harabaz_score(iris_data, kmeans.labels_)
    temp = "iris数据聚" + str(i) + "类calinski_harabaz指数为: " + str(score) + "\n"
    result.append(temp)

with open("data/K-Means模型calinski_harabaz评价.txt", "w") as f:
    for i in range(len(result)):
        f.writelines(result[i])
