# 处理用于模型训练的数据
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
x_traint,y_traint = shuffle(x_train, y_train) # 随机打乱
X_train ,X_test ,Y_train, Y_test = train_test_split(x_traint,y_traint,test_size = 0.2, random_state=30) # 切分数据集到训练集和测试集
from keras.utils import to_categorical # onehot分类
Y_train_binary = to_categorical(Y_train) # 原始为一列0、1分类，处理后变为两列
Y_test_binary = to_categorical(Y_test)

# 建立随机森林回归模型
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor() #回归模型
r = rf.fit(X_train,Y_train_binary) # 训练
acc = r2_score(Y_test_binary, rf.predict(X_test)) #预测并计算准确率
print(acc)

# 建立随机森林分类模型1
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X_train,Y_train_binary)
y_predprob = rf0.predict_proba(X_train)
print("AUC Score (Train): %f"%(metrics.roc_auc_score(Y_train_binary,y_predprob[1]))) # 使用矩阵统计ROC准确率

# 网格搜索调参
from sklearn.model_selection import GridSearchCV
param_test1= {'n_estimators':range(30,70,10)} # n_estimators这个参数，取值从30到70隔10个取一次
gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,              # 新建一个网格搜索器GridSearchCV
                       min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), # estimator参数设定成随机森林分类器
                       param_grid =param_test1, scoring='roc_auc',cv=5)  # 搜索参数为param_test1，评分是roc，交叉验证参数cv默认None
gsearch1.fit(X_train,Y_train_binary)
means = gsearch1.cv_results_['mean_test_score'] # 提取出测试均分
params = gsearch1.cv_results_['params']  # 提取出参数取值
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param)) # 输出例如 0.769062  with:   {'n_estimators': 30}

# 参数重要度
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
np.random.seed(2019)
scores = defaultdict(list) # 存放对应分数
for i in list(X_train): # X_train.shape[1] 是特征数
    X_t = X_test.copy()
    noise = np.random.normal(0,0.5,len(X_test))
    X_t[i] = X_t[i] + noise # 在单个特征维度上加噪声
    
    shuff_acc = r2_score(Y_test_binary, rf.predict(X_t)) # 计算加噪声以后的数据预测结果得出绕乱后的准确率
    scores[i].append((acc-shuff_acc)/acc) # 用准确率差值比作为特征重要度分数
print ("Features sorted by their score:")
print (sorted([(round(np.mean(score), 4), feat) for          # 对分数求均值并取到4位数，再进行排序
              feat, score in scores.items()], reverse=True)) 

