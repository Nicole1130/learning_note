import pandas as pd
import matplotlib.pyplot as plt


#
pandas数据处理————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
df = pd.read_csv("data/AAPL.csv")
df.tail() # 最后五行
df.tail(n) # 最后n行
df.head() # 头5行

df['Close'].max() # 'Close'列的最大值,同理.mean()均值
df['High'].plot() # 画high列
df[['Close', 'Adjusted Close']].plot() # 在一个图中画两列,需要加两个中括号

start_date = '2010-01-22'
end_date = '2010-01-26'
dates = pd.date_range(start_date,end_date) # 通过时间索引截取数据
df1 = pd.DataFrame(index=dates) # 使用dates作为索引值创建空df
dfSPY = pd.read_csv("data/SPY.csv",index_col="Date",parse_dates=True,usecols=['Date','Adj Close'],na_values=['nan']) # 以"date"列作为行索引,
        # parse_dates=True会将"date"列时间数据转为固定格式,usecols只取出'Date','Adj Close'两列,na_values=['nan']将空缺值赋值为nan
dfSPY = dfSPY.rename(columns={'Adj Close':'SPY'}) # 将Adj Close列重命名为SPY
df1 = df1.join(dfSPY,how='inner') #将df1和dfSPY整合到一起,how='inner'表示两者行索引不一致时取交集,同理并集用outer
df.ix[start_index:end_index,columns] #用ix定位截取行列索引序号
symbols.insert(0,'SPY') #列表 insert() 方法将指定对象插入到列表中的指定位置。
df = df.dropna(subset=["SPY"]) #丢掉SPY中为na的那些行。另一个参数，how='all'，代表丢掉一行全为na的，'any'代表一行中有一个为na就丢掉。

df/df.ix[0,:] # 用第一行数据进行归一化

x_all = pd.DataFrame(columns=('id','gender','age','level1','aum227','pos1')) # 新建空dataframe，定义列名
f1_ser = pd.Series(f1) # 将列表list f1转为Series
f1_ser.to_csv('f1.csv') # 存csv文档
x_all.index = x_all.index+1 # 对索引操作
sz_id = sz_detail['sz_id'].copy() # 深拷贝，新开辟内存并存储值，不会对原数据进行修改
chars = set(g2_cod) # 直接获得字符种类集合（不重复）
chars = g2_cod.value_counts(ascending=True) # 统计g2_cod中各元素名称和出现次数并排序，ascending=True升序
g2_cod_dict = dict((c,i/361) for i,c in enumerate(chars.index)) # chars.index是一系列各元素名称，使用enumerate遍历出的i是各位置，c是各名称
                                                            # 将名称作为key，i/361（因为该例中序列总长为361，进行归一化）作为value制成字典
rank_date.index = rank_date['prt_dt'] # 使用日期做索引
rank_date = rank_date.sort_index() #按照索引排序
data = rank_date.truncate(after='2019-02-27') # 截取2019-02-27日之前的信息
id_list = df_user['id'].tolist() #序列Series转list
x_train_detail.drop(columns=['Unnamed: 0'],inplace=True) # 删除指定列，inplace=True 直接对原表进行修改
gender_num = gender.map(lambda x: 0 if x=="M" else 1) # map：对gender中每一个元素操作。如果该元素x=‘M’则返回为0，否则返回1
x = x_all.isnull().sum().sum() # null值统计处理
x_all = x_all.fillna(-1)  # 用-1填充null值
y_1_large1 = pd.concat([y_1,y_1_noise],ignore_index=True) # 将y_1,y_1_noise连接，直接按行添加，忽略索引。
print("预测点击率：",len(data_my.loc[data_my['score']>0.5])/len(data_my)) # data_my['score']>0.5：返回一列，对应score列中大于0.5的位置为true
                                                                         # 其余是false。data_my.loc选择出为true的行

#
numpy数据处理————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
import numpy as np
np.array([1,2,3]) # 生成一维array，设置数据时用【】设置一个（行）维度
np.array([(1,2,3),(4,5,6)]) # 生成二维，用【】定义每一行，用()定义每一列
np.empty(5) # 生成一个一维的零矩阵，但是实际生成的矩阵不是0，而是接近0的随机数
np.ones((5,4),dtype=np.int) # 生成5行4列的全为1的矩阵，设置数类型为int型
np.random.random((3,3)) # 生成3*3的随机数，0-1之间
np.random.randint(0,10,size=5) # 从0-10随机生成5个数
a = np.random.normal(50,10,size=(2,3)) # 生成均值为50，方差为10的服从高斯分布的随机数。
a.shape # 返回一个元祖（行数，列数）
len(a.shape) # 返回维度数.例如(2,3)两行三列，具有两个维度，所以返回2.
a.shape[0] # 返回行数，同理[1]返回列数。
a.size # 返回a矩阵的大小，行*列，本例中2x3=6
a.dtype # 返回a矩阵的数据类型
a.sum() # 返回a矩阵全部求和值。sum(axis=0)返回每一列的求和值，axis=1是按行求和。
a.mean() # 返回均值
np.argmax(a) # 返回a矩阵最大值的索引，这个索引为1维索引值，即[(0,0,0),(1,0,0)]，返回的是3.
