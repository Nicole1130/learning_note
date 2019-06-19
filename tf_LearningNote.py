# -*- coding: utf-8 -*-
学习笔记

#
文件重命名 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#  导入模块
import os
#  获取输入路径
path = 'C:/Users/Freeman/Desktop/images/'   #  获取该目录下所有文件，存入列表中
f = os.listdir(path) #  对获取的文件名进行排序,否则是乱序修改
f.sort()
n = 0 
s = str(n)  # 将int转换为string，从0开始
s = s.zfill(6)  # 字符串长度设置，不足左补零
#  遍历修改每一个文件名
for i in f:   
    oldname=path+f[n]#  获取旧文件名（就是路径+文件名）
    newname=path+s+'.jpg'    #  设置新文件名，根据自己的文件名和类型修改
    os.rename(oldname, newname)    #  调用rename()重命名函数
    print(oldname, '------------>', newname)    #  打印修改结果
    #  更新命名字符串  
    n += 1
    s = str(n)
    s = s.zfill(6)





#
opencv视频中截取帧 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	import cv2
	import os
	 
	videos_src_path = r'./video'  # 视频文件夹
	videos_save_path = r'./frame'  # 保存帧图片的路径
	#文件路径一般是d:\test.txt，但本身语言具有转义符例如\n表示换行，此时如果路径中包含已经定义的转义符时就会报错。可以改为“d:\\test.txt”或@“d:\\test.txt”或r"d:\test.txt"


	videos = os.listdir(videos_src_path)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表list型。
	videos = filter(lambda x: x.endswith('mp4'), videos)  # 将mp4文件读进来，可改为avi等格式
	# filter(function, iterable)   function -- 判断函数  iterable -- 可迭代对象
	# lambda表达式，通常是在需要一个函数，但是又不想费神去命名一个函数的场合下使用，也就是指匿名函数。:冒号前是参数，可以有多个，用逗号隔开，冒号右边的返回值
	# x.endswith('mp4')   判断字符串x是否以mp4结尾  是则返回true  否则返回false

	for each_video in videos:
	    frame_count = 1#帧数等于1
	    # 得到每个文件夹的名字, 并指定每一帧的保存路径
	    each_video_name, _ = each_video.split('.')  # 按'.'分割  例'www.google.com'=>'www', 'google', 'com'
	    os.mkdir(videos_save_path + '/' + each_video_name)  # 创建文件夹目录 例videos_save_path/www/google/com
	    each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'
	    # 合并目录得到保存帧图片的完整路径
	 
	    # 得到完整的视频路径
	    each_video_full_path = os.path.join(videos_src_path, each_video)
	    # 用OpenCV一帧一帧读取出来
	    cap = cv2.VideoCapture(each_video_full_path)
	    
	    success = True
	    while(success):
	        success, frame = cap.read()  #cap.read()按帧读取视频，frame就是每一帧的图像，是个三维矩阵
	        #利用视频对象的read()函数读取视频的某帧，并显示
	        print('Read a new frame: ', success)
	        params = []  
	        params.append(1)
	        # if len(frame) > 0:
	        # if success:
	        if frame is not None and frame_count % 20 == 0:  # 每20帧取一帧图片保存下来
	            cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame, params)#用opencv保存图片，参数分别是文件名 图片 帧
	            print(frame_count)
	        frame_count = frame_count + 1#帧数+1
	    cap.release()#销毁所有窗口，释放资源


#
读取文档—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

	url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
	raw_data = urllib.request.urlopen(url)
	dataset = np.loadtxt(raw_data, delimiter=",")
	#从网络加载数据
	data1=sp.genfromtxt("E:\\stock\\999999month.csv",delimiter=',')
	#本地文档，scipy的文件读取函数genfromtxt，文件路径用\\间隔不然会错误识别为标识符，delimiter标明分隔符。
	#这种读取方法会统一数据类型，例如float64，会把时间数据19901231读成1.99012e+07
	data2=sp.genfromtxt("E:\\stock\\999999month.csv",dtype=[int,float,float,float,float],delimiter=',')
	#增加dtype分割类型读取以后，虽然能读全，但是会变为array field会将每列分为不同空间，f0、f1……，目前不知怎么调用
	pddata1=pd.read_csv("E:\\stock\\999999month.csv")
	#pandas读取函数方便很多，自动识别数据类型，还能为每列标名称以便提取，但是一点，它会自动将第一行默认为每列名称

	filepath1 = './VOC2007/Annotations'
	#例如.py文档的目录是'E:/Damage detection/my/location/text_detection_faster-rcnn/text-detection-ctpn-master/data/VOCdevkit2007'
	#该文档目录是'E:/Damage detection/my/location/text_detection_faster-rcnn/text-detection-ctpn-master/data/VOCdevkit2007/VOC2007/Annotations'
	#即同级目录下，用./找到他们公共路径，从而定位相对路径
	filepath2 = '../VOC2008'
	#该文档目录是'E:/Damage detection/my/location/text_detection_faster-rcnn/text-detection-ctpn-master/data/VOCdevkit2008'
	#即向上跳两级目录，用../找到，从而定位相对路径

	ftrainAll = open('txt/trainAll.txt', 'w')#新建或打开名为trainAll.txt的文档，方式为w写入
	ftrainAll.write("hello nic") #写入文档
	ftrainAll.close() #关闭文档

#
定义函数—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

	def test(*args, **kwargs):
	#定义函数test，*args表示任何多个无名参数，**kwargs表示关键字参数
	    print 'args = ', args
	    print 'kwargs = ', kwargs

	test('a', 1, None, a=1, b='2', c=3)
	#调用函数。结果输出args =  ('a', 1, None) ，kwargs =  {'a': 1, 'c': 3, 'b': '2'} 

#
定义类———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

	class Net:
	    def __init__(self, opt, **kwargs):
	    #self只有在类的方法中才会有，表示类的实例，不需要从外传入数据。
	        self.x = tf.placeholder(tf.float32, [None, 1])
	        #Net类自己的x，和函数外变量无关。
	        self.y = tf.placeholder(tf.float32, [None, 1])
	        l = tf.layers.dense(self.x, 20, tf.nn.relu)
	        #tf.layers.dense是全连接层，输入为self.x ，20个输出，tf.nn.relu是激活函数。
	        out = tf.layers.dense(l, 1)
	        #20个输入（l的个数），一个输出。
	        self.loss = tf.losses.mean_squared_error(self.y, out)
	        #比较self.y和out均方差损失。
	        self.train = opt(0.01, **kwargs).minimize(self.loss)
	        #从传过来的opt的优化函数（学习率为0.01，若有传来的参数则用**kwargs带入）最小化loss
			net_Momentum    = Net(tf.train.MomentumOptimizer, momentum=0.9)
			#使用net类，传入opt=tf.train.MomentumOptimizer，**kwargs参数为momentum=0.9
			net_RMSprop     = Net(tf.train.RMSPropOptimizer)
			#谷歌的AlphaGo使用的就是这个函数
			net_Adam        = Net(tf.train.AdamOptimizer)
			#效果还不错。
			losses_his = [[], [], []]   
			#定义整合所有loss的数组
			for step in range(300):          # 随机索引
			    index = np.random.randint(0, x.shape[0], BATCH_SIZE)
			    b_x = x[index]
			    b_y = y[index]
				for net, l_his in zip(nets, losses_his):
					#zip()函数来可以把列表合并,例如a[1,2]，b[3,4]，zip（a,b）=[(1,3)(2,4)]
				    _, l = sess.run([net.train, net.loss], {net.x: b_x, net.y: b_y})
				    #run两个函数，第一个train函数没有（不需要？）返回值，用_表示，loss函数的返回值付给l。使用类内函数需要加类名，传递的类内参数xy需要加类名。
				    l_his.append(l)     
				    # 将l依次添加到l_his中。
    
#
选取有用数据—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

	x1=pddata2[['time','open']]
	#可以按列名称任意提取组合
	x2=np.array(pddata2['open'])
	#可以先提取出来在用np的array变为数组
	x3 = dataset[:,0:7]
	#行列用逗号区分，逗号前为行范围后为列。x：y表示从x到y。此例行范围为负无穷到正无穷，即每行。列为0至第6列，左闭右开，不包含第七列
	x4 = dataset[:,8]
	#每行第八列
	x5=dataset[:, [1, 3]]
	#每行第一列和第三列组成新组合
	x4 = dataset[::-1]
	#将数据倒叙取出.但若是以pd直接按名称提取出来的值，因为连同序号一起反转了，序号和数据对应关系还是没变，所以后期数据处理与不反转相同。
	#可以先使用np.array提取为数组。
	X_train,X_test, y_train, y_test =train_test_split(train_data,train_target,test_size=0.4, random_state=0)
	#sklearn的随机划分训练集和测试集函数。train_data：所要划分的样本特征集，train_target：所要划分的样本结果，test_size：样本占比。
	npx_train, npx_test = np.split(npx, [800])  
	#np的分割方法，本例npx[1000,1]，npx_train为[800,1], npx_test为[200,1]
	actions_value1 = np.delete(actions_value,1,axis=1)
	#删除actions_value中第一列。例如actions_value=[0,1,2]，1行3列，np.delete(actions_value,1)则删掉了第2列（1），剩下元素按2行一列排序

	list = range(100) #生成一个随机0-100的list
	tv = int(100 * 0.7) 
	trainval = random.sample(list, tv) #随机在list中抽取100*0.7=70个数

#
数据处理—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

	xx=round(x, 4)
	#将x四舍五入，保留小数点后四位小数
	x = np.arange(0，y.shape)
	#range(start, stop, step)，python的range和np的arange相似，生成序列，后者功能更全，可用做向量，步长可为小数。
	X = np.random.randn(2, 10, 8)
	#随机产生格式为（2,10,8）的矩阵，（均值为0，方差为1？）
	np.random.seed(2017) 
	#产生随机数，使用相同的seed种子值（2017），则每次生成的随机数都相同
	x_data = np.linspace(-1, 1, 300, dtype=np.float32)
	#从-1到1线性产生300个数据
	noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
	#随机产生，均值0，方差0.05,大小维度与x_data相同，类型为np.float32
	WInitial = tf.truncated_normal(shape, stddev=0.1)
	#从截断的正态分布中输出随机值，如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择，stddev: 正态分布的标准差
	index = np.random.randint(0, 100, 32)
	#从0到100范围内，随机产生32个整数。
	npx = np.random.uniform(-1, 1, (1000, 1))  
	#uniform从一个均匀分布[-1,1)中随机采样，格式为1000行*1列
	n_data = np.ones((100, 2))
	#100行2列，每个都为1的矩阵。同理 np.zeros()为0矩阵 
	x0 = np.random.normal(2*n_data, 1)     
	#随机产生，每个位置（大小维度与n_data相同）均值都为n_data的两倍，方差为1的矩阵。
	x = np.vstack((x0, x1)) 
	#将x0和x1纵向连接，即每列先把x0数据放完再放x1数据，相当于把x1垒在x0下面。同理横向np.hstack
	normalize_data=(data-np.mean(data))/np.std(data)  
	#0均值标准化，np.std(a, axis=1) 计算每一行的标准差 ，列则用axis=0 
	normalize_data=normalize_data[:,np.newaxis]       
	#增加维度。a[np.newaxis,:]是在行上加维度，即将原本的一维行向量变为列向量，例如[1 2 3 4 5]变为[[1 2 3 4 5]]。
	#a[:,np.newaxis]在列上加维度，即将原本一维行向量中每一个元素分开变为一个个列向量。例如[1 2 3 4 5]变为[[1][2][3][4][5]]。
	xx=x.ravel()
	#ravel函数将x这个多维数据铺平成一维，与flatten()函数的功能一样，但区别在于后者返回的是一个拷贝量，对拷贝量操作不影响原矩阵，而前者是直接影响。
	y_sigmoid = tf.nn.sigmoid(x)
	#对x取sigmoid
	x_image = tf.reshape(xs, [-1, 28, 28, 1])
	#对图像数据重组型，xs是原始图像数据，-1表示暂不考虑xs的个数，[28,28]表示将图像长宽尺寸，1表示通道（灰度图是1，rgb是3）
	action = np.argmax(actions_value1）
	#np.argmax取最大值所在位置

	numpy.clip(a, a_min, a_max, out=None)
	#a是一个数组，后面分别表示最小和最大值.clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
	
	from sklearn.preprocessing import minmaxscaler
	x=minmaxscaler（x）
	#区间放缩到0-1之间
	y=LabelBinarizer().fit_transform(y)
	#将n分类的一维标签数据转为n*n的仅在标签归属位置为1其他位置为0的矩阵
	from sklearn.preprocessing import StandardScaler
	ss_X=StandardScaler()
	#初始化对特征和目标值的标准化器
	x_train = ss_x.fit_transform(x_train)  
	#进行标准化处理
	dataset = Dataset.from_tensor_slices((tfx, tfy))
	#tf新apidataset。将tfx和tfy切片，此处tfx和tfy都是占位符，即将其一一对应，变为（tfx1，tfy2）……形式
	dataset = dataset.shuffle(buffersize=1000).batch(32).repeat(10)
	#在1000为一个epoch，每个epoch内将图片打乱组成大小为32的batch，并重复10次。
	iterator = dataset.make_initializable_iterator() 
	#创建一个 Iterator 枚举这个数据集的元素。
	bx, by = iterator.get_next()  
	#bx,by此处不是具体数值，类似于占位符，get_next()表示从iterator中取下一个。
	sess.run([iterator.initializer, tf.global_variables_initializer()], feed_dict={tfx: npx_train, tfy: npy_train})
	#iterator需要初始run一下，同时把变量初始，feed字典使用的是tfx和tfy，因为作为第一层占位符，只有赋值才能给bx和by使用。

#
图像显示—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	import 
	fig = plt.figure(1)
	#建一张图表1，当只有一张图时可不用该函数。
	plt.ion()
	#开启交互模式
	plt.ioff()
	#关闭交互模式，在plt.show()前使用
	plt.subplot(221)
	#显示图在图组2行2列的第一个位置
	plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
	#散点图，分别x的第一列和第二列，c=y用不同颜色区别两组数据（若是c=‘y’则表示颜色用yellow），s表示点的圆心大小（默认20），
	#lw表示画点的线宽（圆的外圈），cmap='RdYlGn'表示颜色渐变，从红rd到绿gn
	plt.hist(x, bins=15, range=(-7, 10), color='#FF5733')
	#画x数据的直方图，bins是柱的个数，range是总范围
	plt.show()
	#显示。plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()
	labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
	#多个标注命名
	for i, l_his in enumerate(losses_his):
	#enumerate函数用于遍历序列中的元素以及它们的下标，i是0到losses_his长度的序号
	    plt.plot(l_his, label=labels[i])
	    #画l_his，标注为labels[i]（颜色不一样）
	plt.legend(loc='best')
	#loc参数设置图例的显示位置，loc='best'表示自适应
	plt.xlabel('Steps')
	#横坐标名
	plt.ylabel('Loss')
	plt.yticks(np.linspace(0,20,10,endpoint=True))
	#yticks为y轴的刻度设置颜色、大小、方向,以及标签大小，第一个参数是y轴的位置，第二个参数是具体标签
	#numpy.linspace()方法返回一个等差数列数组，在（0,20）范围内随机取10个数，如果endpoint=True意味着要将20包含在数列里，这时只能取0,2,4……20这10个数
	#但如果endpoint=True=False则20不包含于数组内，此时10个数就是随机等差选的，例如1,3……19
	plt.ylim((0, 0.2))
	#y取值范围是0到0.2
	plt.cla()
	#清空图像区域
	plt.savefig("examples2.jpg")
	#将图片存在根目录下
	from mpl_toolkits.mplot3d import Axes3D
	ax=plt.subplot(111,projection='3d')
    ax.scatter(encoder_result[:, 0], encoder_result[:, 1],encoder_result[:, 2],s=5, c=mnist.test.labels)
    #画encoder_result[:, 0], encoder_result[:, 1],encoder_result[:, 2]的三维图


#
tensorflow———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	x = tf.placeholder(tf.float32, [None, 784])
	#声明语句，占位符placeholder,2维的浮点数张量,None表示此张量的第一个维度可以是任何长度的
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	#声明语句，Variable是变量，每行（784行）权重用来乘每列x（784列），10列对应mnist的10个结果。
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	#.softmax正则化回归，.matmul矩阵乘法运算
	y_ = tf.placeholder("float", [None,10])
	#真实y值

	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	#交叉熵
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) 
	#reduction_indices=[1]参数表示沿着行（1）方向压缩求和，即ys * tf.log(prediction)若是一个矩阵则将每一列的值都
	#数值累加后赋值给每一行的第一列，缩成一个x行一列的列向量，但最后函数输出其实还是整理成了行向量。
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	#梯度下降优化算法求最小化交叉熵，0.01学习率
	init = tf.initialize_all_variables()
	#模型设计好需要初始化所有变量
	sess = tf.Session()
	sess.run(init)
	#sess会话启动模型
	for i in range(1000):
	  batch_xs, batch_ys = mnist.train.next_batch(100)
	  #循环1000次，每循环都随机抓取训练数据中的100个批处理数据点
	  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	  #运行train_step，feed_dict是字典，x从batch_xs中取，y同理
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	#tf.argmax(y,1)返回y中行（axis=1）最大元素的索引值，tf.equal比较两者相同与否，返回的是布尔值。
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	#tf.cast(x, dtype)将x的数据格式转化成dtype，本例T/F变1/0.而.reduce_mean是求均值
	print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

	tf.rank(arr).eval()
	#打印矩阵arr维度
	tf.shape(arr).eval()
	#打印arr的大小
	tf.argmax(arr,0).eval()
	#打印最大的索引，参数0为按列，1为行

	out = tf.nn.dropout(x, keep_prob)
	#在不同的训练过程中随机扔掉一部分神经元，一般用在全连接层，keep_prob设置神经元被选中的概率，在初始化时keep_prob是一个占位符，在run时设置keep_prob具体的值


	init= tf.global_variables_initializer()
	#保存模型需要先把variables初始化
	saver = tf.train.Saver()
	#定义saver
	with tf.Session() as sess:
	   sess.run(init)
	   save_path = saver.save(sess, 'logs/model.ckpt')
	   print("Save to path: ", save_path)
	# 用 saver 将所有的 variable 保存到定义的路径.该保存路径为.py文件所在目录下logs文件夹，存为了名为model的ckpt文件。
	# 调用的时候需要重载saver
	W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
	b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
	#先定义与保存的module中相同维度形状类型和名称的变量。具体数据随便填写，可以用np.arange随即产生，reshape的维度大小和module一定要相同。
	#不需要初始化
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    saver.restore(sess, 'logs/model.ckpt')
	    print("weights:", sess.run(W))
	    print("biases:", sess.run(b))
	# 用 saver 从路径中将 save_net.ckpt 保存的 W 和 b restore 进来

#
tensorboard———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

	#cmd下运行tensorboard --logdir=c:/logs 启动tensorboard用浏览器查看返回的网址（http://127.0.1.1:6006/）。
	# 使用tf.summary.scalar记录标量 
	# 使用tf.summary.histogram记录数据的直方图 
	# 使用tf.summary.distribution记录数据的分布图 
	# 使用tf.summary.image记录图像数据 

	# define placeholder for inputs to network
	# 区别：大框架，里面有 inputs x，y
	with tf.name_scope('inputs'):
	    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
	    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

	l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
	prediction = add_layer(l1, 10, 1, activation_function=None)

	# the error between prediciton and real data
	# 区别：定义框架 loss
	with tf.name_scope('loss'):
	    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

	# 区别：定义框架 train
	with tf.name_scope('train'):
	    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	# 区别：sess.graph 把所有框架加载到一个文件中放到文件夹"logs/"里 
	# 接着打开terminal，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir='logs/'
	# 会返回一个地址，然后用浏览器打开这个地址，在 graph 标签栏下打开
	writer = tf.train.SummaryWriter("logs/", sess.graph)
	# important step
	sess.run(tf.initialize_all_variables())
	merged = tf.summary.merge_all()
	#将sum结合在一起
	sess.run(merged,feed_dict={xs: x_data, ys: y_data})
	#也需要run一下
	writer.add_summary(result, i)
	#.add函数可以补加

	#在训练的过程在参数是不断地在改变和优化的，我们往往想知道每次迭代后参数都做了哪些变化，
	#可以将参数的信息展现在tenorbord上，因此我们专门写一个方法来收录每次的参数信息。
	def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)

	#调用的时候直接调用函数variable_summaries即可。可以多次使用，记录weights，bias等等。例如下。
	weights = weight_variable([input_dim, output_dim])
    variable_summaries(weights)
#
实例：MNIST的CNN网络—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

	#3-5min，跑出来结果趋近于97%。
	from __future__ import print_function
	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	#.read_data_sets函数参数文件名若为'MNIST_data'则在本py文件所在文件夹下建立名为MNIST_data的文件夹；若为"/tmp/data/"则在根文件夹（c、d、e、f盘）下
	#建立一个名为tmp的文件夹，其下再放一个data文件夹存放数据。注意前者为单引号，后者为双引号。

	#将各个标准元件W,b，accuracy和两层网络都先定义为函数，具体使用的时候才调用，省步骤，可广泛借鉴。
	def compute_accuracy(v_xs, v_ys):
	    global prediction
	    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
	    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
	    return result

	def weight_variable(shape):
	    initial = tf.truncated_normal(shape, stddev=0.1)
	    return tf.Variable(initial)

	def bias_variable(shape):
	    initial = tf.constant(0.1, shape=shape)
	    return tf.Variable(initial)

	def conv2d(x, W):
	    # stride [1, x_movement, y_movement, 1]
	    # Must have strides[0] = strides[3] = 1
	    #padding有'same'和‘valid’，后者代表只进行有效的卷积，即对边界数据不处理，前者反之。
	    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
	    # stride [1, x_movement, y_movement, 1]
	    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	# define placeholder for inputs to network
	xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
	ys = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)
	x_image = tf.reshape(xs, [-1, 28, 28, 1])
	# print(x_image.shape)  # [n_samples, 28,28,1]

	## conv1 layer ##
	W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
	h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

	## conv2 layer ##
	W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
	h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

	## fc1 layer ##
	W_fc1 = weight_variable([7*7*64, 1024])
	b_fc1 = bias_variable([1024])
	# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	## fc2 layer ##
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


	# the error between prediction and real data
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
	                                              reduction_indices=[1]))       # loss
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	sess = tf.Session()
	# important step
	# tf.initialize_all_variables() no long valid from
	# 2017-03-02 if using tensorflow >= 0.12
	if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	    init = tf.initialize_all_variables()
	else:
	    init = tf.global_variables_initializer()
	sess.run(init)

	for i in range(1000):
	    batch_xs, batch_ys = mnist.train.next_batch(100)
	    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
	    if i % 50 == 0:
	        print(compute_accuracy(
	            mnist.test.images[:1000], mnist.test.labels[:1000]))

#
实例：MNIST的CNN网络（更新tensorflow版）—————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	import numpy as np
	import matplotlib.pyplot as plt

	tf.set_random_seed(1)
	np.random.seed(1)

	BATCH_SIZE = 50
	LR = 0.001              # learning rate

	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # they has been normalized to range (0,1)
	test_x = mnist.test.images[:2000]
	test_y = mnist.test.labels[:2000]

	# plot one example
	print(mnist.train.images.shape)     # (55000, 28 * 28)
	print(mnist.train.labels.shape)   # (55000, 10)
	plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
	plt.title('%i' % np.argmax(mnist.train.labels[0])); plt.show()

	tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
	image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
	tf_y = tf.placeholder(tf.int32, [None, 10])            # input y

	# CNN
	conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
	    inputs=image,
	    filters=16,
	    kernel_size=5, #卷积核5*5
	    strides=1, #卷积核的步长
	    padding='same',# 扩充边缘
	    activation=tf.nn.relu
	)           # -> (28, 28, 16)
	pool1 = tf.layers.max_pooling2d(
	    conv1,
	    pool_size=2,
	    strides=2,
	)           # -> (14, 14, 16)
	conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
	pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
	flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
	output = tf.layers.dense(flat, 10)              # output layer

	loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
	train_op = tf.train.AdamOptimizer(LR).minimize(loss)

	accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
	    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

	sess = tf.Session()
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
	sess.run(init_op)     # initialize var in graph

	# following function (plot_with_labels) is for visualization, can be ignored if not interested
	from matplotlib import cm
	try: from sklearn.manifold import TSNE; HAS_SK = True
	#sklearn的两个高维数据可视化工具 —— PCA & TSNE（t分布随机邻域嵌入）。SNE保留下的属性信息，更具代表性，也即最能体现样本间的差异，SNE 运行极慢，PCA 则相对较快；
	except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')
	def plot_with_labels(lowDWeights, labels):#自定义的画图函数
	    plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]#lowDWeights的第0列是，第一列是预测值
	    for x, y, s in zip(X, Y, labels):
	        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
	        #颜色c设成彩虹图，因为s是labels，数值在[0,9）,所以具体色度是255rgb*归一化后的标签值
	    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

	plt.ion()
	for step in range(600):
	    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
	    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
	    if step % 50 == 0:
	        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
	        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

	        if HAS_SK:
	            # Visualization of trained flatten layer (T-SNE)
	            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
	            low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
	            #将flat_representation内数据使用tsne.fit_transform函数降维，因为flat_representation行取值[0,500]所以low_dim_embs是（500,2）
	            labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
	plt.ioff()

	# print 10 predictions from test data
	test_output = sess.run(output, {tf_x: test_x[:10]})
	pred_y = np.argmax(test_output, 1)
	print(pred_y, 'prediction number')
	print(np.argmax(test_y[:10], 1), 'real number')

#
实例：MNIST的RNN网络—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

    #比前一个CNN速度快，且测试正确率为98%
    #RNN在同个workspace（软件不重启）中重复使用两次及以上、或者同一个rnn网络中训练和测试在同一个文件中，所有涉及到重用模型的时候都会报错。因为模型参数已经固定。
    #用with tf.variable_scope（name）定义域区分。例如
    #with tf.variable_scope('lstm'，reuse=True):
    #	outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
    #若重复运行一个文件，在软件不重启的情况下，每次把名字设为不一样的即可。
    #训练测试同文件的话，用with tf.variable_scope('lstm'，reuse=True)，名字设相同（因为用的是同个medel参数），训练时reuse=False，测试为True（意为确认调用先前参数）
    #或用    
    #with tf.variable_scope('rnn') as scope:
    #    sess = tf.Session()
    #    train_rnn2 = RNN(train_config)
    #    scope.reuse_variables()   //声明下面的rnn模型不是误操作而是想要重新调用train_rnn2训练出的rnn结构给test使用
    #    test_rnn2 = RNN(test_config)
	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data

	# set random seed for comparing the two result calculations
	tf.set_random_seed(1)
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	#超参数
	lr = 0.001
	training_iters = 100000 #train step的上限
	batch_size = 128

	n_inputs = 28   # MNIST data input (img shape: 28*28)
	n_steps = 28    # time steps
	n_hidden_units = 128   # neurons in hidden layer
	n_classes = 10      # MNIST classes (0-9 digits)

	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
	y = tf.placeholder(tf.float32, [None, n_classes])

	# 相当于定义了两个隐藏层，进lstm的cell前、后分别使用。
	weights = {
	    # (28, 128)
	    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
	    # (128, 10)
	    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
	}
	biases = {
	    # (128, )
	    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
	    # (10, )
	    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
	}


	def RNN(X, weights, biases):
	    #输入X的batch为128个（128张图片），28个时间步（一张图28行），28个输入（一行28个像素）
	    #输入cell前先经过一个隐藏层，即‘in’的wx+b，因为w和b定义的是二维数组，想要拿x和w相乘，需要想把x变为二维，X ==> (128 batch * 28 steps, 28 inputs)
	    X = tf.reshape(X, [-1, n_inputs])
	    # 将batch和steps联合成一个维度，多大暂时不管（-1），第二个维度的input尺寸不变。现在可与参数运算
	    X_in = tf.matmul(X, weights['in']) + biases['in']
	    #因为输入rnncell的数据还得是3维，所以再变回来。定了第二和第三维尺寸，第一维大小可省略
	    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

	    # 定义lstm的cell，使用basic LSTM Cell.
	    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
	        #forget_bias=1.0遗忘门偏置，开始设为1因为我们一开始不想它遗忘任何信息，后期参数自动更新， state_is_tuple官方建议为True
	    else:
	        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
	        #新版本参数默认。
	    # cell初始化
	    init_state = cell.zero_state(batch_size, dtype=tf.float32)

	    # 两种cnn形式tf.nn.rnn（已经淘汰）和tf.nn.dynamic_rnn.后者能接收(batch, steps, inputs)或(steps, batch, inputs)作为X_in
	    # 若inputs 为 (batches, steps, inputs) ==> time_major=False反之True;输出两个参数，output和final_state（lstm中final_state包含两项c_state和h_state）
	    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

	    # 将输出再过一层隐藏层。
	    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    
	    else:
	        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
	        #outputs形式与输入一样是[batch,step,hidden]，我们想要一张图（一个batch）的一个最终结果（最后一个hidden）所以要将其变成[step,batch,hidden]
			#[1,0,2] 是告诉 tf 要如何翻转现有的三维张量, 假设原有的张量是 [0,1,2] 的维度顺序, 使用 tf.transpose, 会将[0,1,2] 的0 和 1 维数据互换维度.
	        #再unstack（矩阵分解，例如[[3 2 4 5 6][1 6 7 8 0]]拆成[array([3, 2, 4, 5, 6]), array([1, 6, 7, 8, 0])]）
	        #此时outputs的最后一个，128*28（[batch,hidden]）的矩阵便为last_step的结果，索引是[-1]
	    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  
	    #调用最后一个 outputs (在这个例子中,和final_state[1]（1也就是h_state）是一样的)，载入rnn的输出隐藏层。

	    return results


	pred = RNN(x, weights, biases)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	train_op = tf.train.AdamOptimizer(lr).minimize(cost)

	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	with tf.Session() as sess:
	    # tf.initialize_all_variables() no long valid from
	    # 2017-03-02 if using tensorflow >= 0.12
	    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	        init = tf.initialize_all_variables()
	    else:
	        init = tf.global_variables_initializer()
	    sess.run(init)
	    step = 0
	    while step * batch_size < training_iters:
	        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
	        sess.run([train_op], feed_dict={
	            x: batch_xs,
	            y: batch_ys,
	        })
	        if step % 20 == 0:
	            print(sess.run(accuracy, feed_dict={
	            x: batch_xs,
	            y: batch_ys,
	            }))
	        step += 

#
实验1.用lstm回归股票月线—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

	#其实是对月线图型训练好多次的过拟合结果
	#但模型可借鉴
	from sklearn import preprocessing
	import tensorflow as tf
	import numpy as np

	all= np.loadtxt("E:\\stock\\999999month.csv",delimiter=",",skiprows=0)
	x=all[:,[0,1,2,4]]
	y=all[:,3]
	train_y=np.reshape(y,[-1,1])

	ss_x = preprocessing.StandardScaler()
	train_x = ss_x.fit_transform(x)
	ss_y = preprocessing.StandardScaler()
	train_y = ss_y.fit_transform(train_y)
	#
	BATCH_START = 0     # 建立 batch data 时候的 index
	TIME_STEPS = 17     # 时间步长，也理解为我们认为17天内数据有相关性
	BATCH_SIZE = 19		# 一次放入19个17天数据，设置batch能加速训练速度
	INPUT_SIZE = 4      # 输入4个变量。
	OUTPUT_SIZE = 1     # 输出当日收盘价
	CELL_SIZE = 10      # RNN 的 hidden unit size
	LR = 0.006          # learning rate
	x_part1 = train_x[BATCH_START,0]
	#
	def get_batch_boston():
	    global train_x, train_y,BATCH_START, TIME_STEPS
	    x_part1 = train_x[BATCH_START : BATCH_START+TIME_STEPS*BATCH_SIZE]
	    #BATCH_START是0，TIME_STEPS*BATCH_SIZ正好323（刻意设计的），也就是说一次把所有数据都放进去训练 
	    y_part1 = train_y[BATCH_START : BATCH_START+TIME_STEPS*BATCH_SIZE]
	    print('时间段=', BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)

	    seq =x_part1.reshape((BATCH_SIZE, TIME_STEPS ,INPUT_SIZE))
	    res =y_part1.reshape((BATCH_SIZE, TIME_STEPS ,1))

	    BATCH_START += TIME_STEPS

	    return [seq , res  ]
	#
	class LSTMRNN(object):
	#直接将网络定义为一个类，很漂亮
	    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
	        self.n_steps = n_steps
	        self.input_size = input_size
	        self.output_size = output_size
	        self.cell_size = cell_size
	        self.batch_size = batch_size
	        with tf.name_scope('inputs'):
	            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
	            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
	        with tf.variable_scope('in_hidden'):
	            self.add_input_layer()
	        with tf.variable_scope('LSTM_cell'):
	            self.add_cell()
	        with tf.variable_scope('out_hidden'):
	            self.add_output_layer()
	        with tf.name_scope('cost'):
	            self.compute_cost()
	        with tf.name_scope('train'):
	            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
	                                                  
	    #增加一个输入层
	    def add_input_layer(self,):
	        # l_in_x:(batch*n_step, in_size),相当于把这个批次的样本串到一个长度17的时间线上，每批次19个样本
	        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  #-1 表示任意行数
	        # Ws (in_size, cell_size)
	        Ws_in = self._weight_variable([self.input_size, self.cell_size])
	        # bs (cell_size, )
	        bs_in = self._bias_variable([self.cell_size,])
	        # l_in_y = (batch * n_steps, cell_size)
	        with tf.name_scope('Wx_plus_b'):
	            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
	        # reshape l_in_y ==> (batch, n_steps, cell_size)
	        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

	    #多时刻的状态叠加层
	    def add_cell(self):
	        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
	        with tf.name_scope('initial_state'):
	            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
	        #time_major=False 表示时间主线不是第一列batch
	        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
	            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

	    # 增加一个输出层
	    def add_output_layer(self):
	        # shape = (batch * steps, cell_size)
	        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
	        Ws_out = self._weight_variable([self.cell_size, self.output_size])
	        bs_out = self._bias_variable([self.output_size, ])
	        # shape = (batch * steps, output_size)
	        with tf.name_scope('Wx_plus_b'):
	            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out #预测结果

	    def compute_cost(self):
	        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
	            [tf.reshape(self.pred, [-1], name='reshape_pred')],
	            [tf.reshape(self.ys, [-1], name='reshape_target')],
	            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
	            average_across_timesteps=True,
	            softmax_loss_function=self.ms_error,
	            name='losses'
	        )
	        #.sequence_loss_by_example是逐个计算实际值和预测值的误差再求和，ms_error是自己定义的。
	        with tf.name_scope('average_cost'):
	            self.cost = tf.div(
	                tf.reduce_sum(losses, name='losses_sum'),
	                self.batch_size,
	                name='average_cost')
	            tf.summary.scalar('cost', self.cost)

	    @staticmethod
	    #定义了静态方法，它无法访问类属性、实例属性，相当于一个相对独立的方法，跟类其实没什么关系，换个角度来讲，其实就是放在一个类的作用域里的函数而已。
		#还有一种classmethod，类成员方法，也同样无法访问实例变量，但可以访问类变量
	    def ms_error(labels, logits):
	        return tf.square(tf.subtract(labels, logits))

	    def _weight_variable(self, shape, name='weights'):
	        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
	        return tf.get_variable(shape=shape, initializer=initializer, name=name)

	    def _bias_variable(self, shape, name='biases'):
	        initializer = tf.constant_initializer(0.1)
	        return tf.get_variable(name=name, shape=shape, initializer=initializer)



	if __name__ == '__main__':
	   # seq, res  = get_batch_boston()
	    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
	    sess = tf.Session()
	    merged = tf.summary.merge_all()
	    writer = tf.summary.FileWriter("logs", sess.graph)
	    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	        init = tf.initialize_all_variables()
	    else:
	        init = tf.global_variables_initializer()
	    sess.run(init)

	    pred_res=None
	    for i in range(20):
	        seq, res = get_batch_boston()
	        if i == 0:
	            feed_dict = {
	                    model.xs: seq,
	                    model.ys: res,
	                    # create initial state
	            }
	        else:
	            feed_dict = {
	                model.xs: seq,
	                model.ys: res,
	                model.cell_init_state: state    # use last state as the initial state for this run
	            }

	        _, cost, state, pred = sess.run(
	            [model.train_op, model.cost, model.cell_final_state, model.pred],
	            feed_dict=feed_dict)
	        pred_res=pred


	        result = sess.run(merged, feed_dict)
	        writer.add_summary(result, i)
	        print('第{0}次， cost: '.format(i+1), round(cost, 4))
	        #使用.format函数能将{0}替换为i+1，用round将cost四舍五入并保留小数点后4位。
	        BATCH_START=0 #从头再来一遍

	    ###画图###########################################################################
	    import matplotlib.pyplot as plt
	    fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
	    axes = fig.add_subplot(1, 1, 1)
	    line1,=axes.plot(range(323), pred.flatten()[-323:] , 'b--',label='prediction')
	    line2,=axes.plot(range(323), train_y.flatten()[ - 323:], 'r',label='target')
	    axes.grid()
	    fig.tight_layout()
	    #图像外部边缘的调整可以使用plt.tight_layout()进行自动控制
	    plt.legend(handles=[line1,  line2])
	    plt.title('rnn')
	    plt.show()

#
实验2.基于实验1对日线分段训练————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

	#在实验1的基础上，改变了get_batch_boston()函数，让它从头依次取20个10天数据放进网络训练（即第一次取（0,200）第二次（1,201）……），希望能学到10天内规律
	#迭代2000次（基本遍历一次全部数据）以后cost便不再更迭，效果很差。
	#改变迭代至20000次、batchsize和cellsize对结果没多大影响。
	#模型没学到东西？？？？？
	from sklearn import preprocessing
	import tensorflow as tf
	import numpy as np

	all= np.loadtxt("E:\\stock\\999999pureday.csv",delimiter=",",skiprows=0)
	#导入的是沪指日线纯数据
	x=all[:,[0,1,2,4]]
	y=all[:,3]
	train_y=np.reshape(y,[-1,1])
	ss_x = preprocessing.StandardScaler()
	train_x = ss_x.fit_transform(x)
	ss_y = preprocessing.StandardScaler()
	train_y = ss_y.fit_transform(train_y)
	#
	BATCH_START = 0      # 建立 batch data 时候的 index
	TIME_STEPS = 10     # 以10天当一个时间段
	BATCH_SIZE = 20    #一次训练10个平行时间段，即10个样本
	INPUT_SIZE = 4      # 输入为开盘、最低、最高、成交量，4个维度
	OUTPUT_SIZE = 1     # 输出为收盘价，1个维度
	CELL_SIZE = 128      # RNN 的 hidden unit size
	LR = 0.006          # learning rate
	index=0
	#
	def get_batch_boston():
	    global train_x, train_y,BATCH_START, TIME_STEPS,index
	    if index+TIME_STEPS*BATCH_SIZE<6575: #索引起始加上一次放进的数据量要小于总数据量
	        index=index
	    else:
	        index=0 #如果超过了就重新从第0个数据开始循环输入
	    x_part1 = train_x[index : index+TIME_STEPS*BATCH_SIZE]
	    #第一次放0-200行，每行4列，800个数据；第二次就放1-201的800个
	    y_part1 = train_y[index : index+TIME_STEPS*BATCH_SIZE]
	    if index % 40 == 0:
	        print('时间段=', index, index+TIME_STEPS*BATCH_SIZE)

	    seq =x_part1.reshape((BATCH_SIZE, TIME_STEPS ,INPUT_SIZE))
	    #将数据整理，好放入rnn网络
	    res =y_part1.reshape((BATCH_SIZE, TIME_STEPS ,1))

	    index+=1
	    return [seq , res]
	#
	class LSTMRNN(object):
	    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
	        self.n_steps = n_steps
	        self.input_size = input_size
	        self.output_size = output_size
	        self.cell_size = cell_size
	        self.batch_size = batch_size
	        with tf.name_scope('inputs'):
	            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
	            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
	        with tf.variable_scope('in_hidden'):
	            self.add_input_layer()
	        with tf.variable_scope('LSTM_cell'):
	            self.add_cell()
	        with tf.variable_scope('out_hidden'):
	            self.add_output_layer()
	        with tf.name_scope('cost'):
	            self.compute_cost()
	        with tf.name_scope('train'):
	            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
	                                                  
	    def add_input_layer(self,):
	        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  #-1 表示任意行数
	        Ws_in = self._weight_variable([self.input_size, self.cell_size])
	        bs_in = self._bias_variable([self.cell_size,])
	        with tf.name_scope('Wx_plus_b'):
	            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
	        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

	    def add_cell(self):
	        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
	        with tf.name_scope('initial_state'):
	            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
	        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
	            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

	    def add_output_layer(self):
	        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
	        Ws_out = self._weight_variable([self.cell_size, self.output_size])
	        bs_out = self._bias_variable([self.output_size, ])
	        with tf.name_scope('Wx_plus_b'):
	            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out #预测结果

	    def compute_cost(self):
	        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
	            [tf.reshape(self.pred, [-1], name='reshape_pred')],
	            [tf.reshape(self.ys, [-1], name='reshape_target')],
	            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
	            average_across_timesteps=True,
	            softmax_loss_function=self.ms_error,
	            name='losses'
	        )
	        with tf.name_scope('average_cost'):
	            self.cost = tf.div(
	                tf.reduce_sum(losses, name='losses_sum'),
	                self.batch_size,
	                name='average_cost')
	            tf.summary.scalar('cost', self.cost)

	    @staticmethod
	    def ms_error(labels, logits):
	        return tf.square(tf.subtract(labels, logits))

	    def _weight_variable(self, shape, name='weights'):
	        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
	        return tf.get_variable(shape=shape, initializer=initializer, name=name)

	    def _bias_variable(self, shape, name='biases'):
	        initializer = tf.constant_initializer(0.1)
	        return tf.get_variable(name=name, shape=shape, initializer=initializer)

	if __name__ == '__main__':
	    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
	    sess = tf.Session()
	    merged = tf.summary.merge_all()
	    writer = tf.summary.FileWriter("logs", sess.graph)
	    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	        init = tf.initialize_all_variables()
	    else:
	        init = tf.global_variables_initializer()
	    sess.run(init)

	    pred_res=None
	    for i in range(20000):
	        #训练20000次
	        seq, res = get_batch_boston()#     有问题！，i和index的作用没体现。
	        if i == 0:
	            feed_dict = {
	                    model.xs: seq,
	                    model.ys: res,
	                    # 第一次训练没有state
	            }
	        else:
	            feed_dict = {
	                model.xs: seq,
	                model.ys: res,
	                model.cell_init_state: state   
	                # 往后使用网络之前的状态
	            }

	        _, cost, state, pred = sess.run(
	            [model.train_op, model.cost, model.cell_final_state, model.pred],
	            feed_dict=feed_dict)
	        pred_res=pred


	        result = sess.run(merged, feed_dict)
	        writer.add_summary(result, i)
	        if i % 150 == 0:
	            print('第{0}次， cost: '.format(i+1), round(cost, 4))

	    import matplotlib.pyplot as plt
	    fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
	    axes = fig.add_subplot(1, 1, 1)
	    line1,=axes.plot(range(30), pred.flatten()[-30:] , 'b--',label='prediction')
	    line2,=axes.plot(range(30), train_y.flatten()[-30:], 'r',label='target')
	    #只画了最后30个点
	    axes.grid()
	    fig.tight_layout()
	    plt.legend(handles=[line1,  line2])
	    plt.title('rnn')
	    plt.show()
	    plt.savefig("examples2.jpg")  

#
实例：MNIST的autoencoder—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

    #1. 显示对比编解码前后的图像
	from __future__ import division, print_function, absolute_import
	import tensorflow as tf
	import numpy as np
	import matplotlib.pyplot as plt
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

	learning_rate = 0.01
	training_epochs = 5
	#一个epoch是指把所有训练数据完整的过一遍，一次epoch=所有训练数据forward+backward后更新参数的过程。
	#而一次iteration迭代=[batch size]个训练数据forward+backward后更新参数过程。
	batch_size = 256
	display_step = 1
	#每一次都显示
	examples_to_show = 10
	#验证的时候用10个显示

	n_input = 784  # MNIST输入，28*28

	X = tf.placeholder("float", [None, n_input])

	# hidden layer settings
	n_hidden_1 = 256 # 1st layer num features
	n_hidden_2 = 128 # 2nd layer num features
	weights = {
	    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
	}
	biases = {
	    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
	}

	def encoder(x):
	    # Encoder Hidden layer with sigmoid activation #1
	    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
	    # Decoder Hidden layer with sigmoid activation #2
	    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
	    return layer_2

	def decoder(x):
		#这里传来的参数虽然名字也定义为了x，但是不同于encoder的x是原始图片信息，这里的x是encoder函数编码后的信息
	    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
	    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
	    return layer_2

	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)

	y_pred = decoder_op
	y_true = X
	#预测值是decoder传出的信息，而真实值是传入图片信息x

	cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
	#cost function是最小二乘，.pow（x,n）表示计算x的n次方
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	# Launch the graph
	with tf.Session() as sess:
	    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	        init = tf.initialize_all_variables()
	    else:
	        init = tf.global_variables_initializer()
	    sess.run(init)
	    total_batch = int(mnist.train.num_examples/batch_size)
	    for epoch in range(training_epochs):
	        for i in range(total_batch):
	            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
	            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
	        if epoch % display_step == 0:
	            print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
	    print("Optimization Finished!")

	    # 显示encoder的输出结果，与原始图像比较
	    encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
	    f, a = plt.subplots(2, 10, figsize=(10, 2))
	    #定义一个2行10列的画板（f的存在意义？）
	    for i in range(examples_to_show):
	        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
	        #a的第一行的第i个显示第i个原始图像
	        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
	        #a的第二行的第i个显示encoder输出结果
	    plt.show()



	#2. 显示编码后聚类情况
	from __future__ import division, print_function, absolute_import
	import tensorflow as tf
	import numpy as np
	import matplotlib.pyplot as plt
	from tensorflow.examples.tutorials.mnist import input_data

	mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
	learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
	training_epochs = 10
	batch_size = 256
	display_step = 1
	# Network Parameters
	n_input = 784  # MNIST data input (img shape: 28*28)
	# tf Graph input (only pictures)
	X = tf.placeholder("float", [None, n_input])
	# hidden layer settings
	n_hidden_1 = 128
	n_hidden_2 = 64
	n_hidden_3 = 10
	n_hidden_4 = 2
	weights = {
	    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
	    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
	    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
	    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),
	    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
	    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
	    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
	    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
	}
	biases = {
	    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
	    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
	    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
	    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
	    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
	}
	def encoder(x):
	    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
	    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
	    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),biases['encoder_b3']))
	    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),biases['encoder_b4'])
	    #希望最后返回的数据不是（0,1）范围，而是无穷范围内的值，故不使用sigmoid而是直接相加
	    return layer_4
	def decoder(x):
	    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
	    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
	    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),biases['decoder_b3']))
	    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),biases['decoder_b4']))
	    return layer_4

	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)

	y_pred = decoder_op
	y_true = X

	cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


	# Launch the graph
	with tf.Session() as sess:
	    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	        init = tf.initialize_all_variables()
	    else:
	        init = tf.global_variables_initializer()
	    sess.run(init)
	    total_batch = int(mnist.train.num_examples/batch_size)
	    for epoch in range(training_epochs):
	        for i in range(total_batch):
	            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
	            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
	        if epoch % display_step == 0:
	            print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
	    print("Optimization Finished!")

		encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
	    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
	    plt.colorbar()
	    plt.show()

#
实例：Batch Normalization ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	import numpy as np
	import tensorflow as tf
	import matplotlib.pyplot as plt

	ACTIVATION = tf.nn.relu# 每一层都使用 relu 
	N_LAYERS = 7 # 一共7层隐藏层
	N_HIDDEN_UNITS = 30# 每个层隐藏层有 30 个神经元

	def fix_seed(seed=1):
	#固定随机序号
	#python的函数定义可以直接在参数里赋值
	    np.random.seed(seed)
	    tf.set_random_seed(seed)


	def plot_his(inputs, inputs_norm):
	# 画每层输入的直方图
	    for j, all_inputs in enumerate([inputs, inputs_norm]):
	    #all_inputs是input里的x数据，j是位置下标
	        for i, input in enumerate(all_inputs):
	            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
	            #画2行，数据长度列的子视图，分别放在j*len(all_inputs)+(i+1)位置上
	            #j*len(all_inputs)+(i+1)当j=0，i+1肯定不会超过一行的长度len(all_inputs)，故全部j=0的都在第一行，j=1的在第二行
	            plt.cla()
	            if i == 0:
	                the_range = (-7, 10)
	            #如果是第一个就把坐标范围限定在（-7，10），因为第一个是第一层的输入
	            else:
	                the_range = (-1, 1)
	            #其他层因为归一化了，所以范围在（-1,1）    
	            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
	            #将input铺成一维，画15柱的直方图，x坐标范围就是以上定义的范围
	            plt.yticks(())
	            if j == 1:
	                plt.xticks(the_range)
	            #如果j是第二个，x轴的刻度范围就是the_range，反之就是自适应（？）
	            else:
	                plt.xticks(())
	            ax = plt.gca()
	            #当前的图表和子图可以使用plt.gcf()和plt.gca()获得
	            ax.spines['right'].set_color('none')
	            ax.spines['top'].set_color('none')
	            #把右边和上边的边界（坐标框线）设置为不可见  
	        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
	        #如果j=0，title就显示Without normalizing，反之With normalizing。即第一行的title是前者，第二行是后者
	    plt.draw()
	    plt.pause(0.01)


	def built_net(xs, ys, norm):
	#函数被调用以后不立即执行add_layer，而是等到被调用的时候才执行
	    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):

	        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
	        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	        Wx_plus_b = tf.matmul(inputs, Weights) + biases

	        if norm:
	        #用norm这个参数表示是否进行BN。函数默认为false，若传入norm为true则运行以下函数
	            fc_mean, fc_var = tf.nn.moments(Wx_plus_b,axes=[0],)
	            #tf.nn.moments函数计算Wx_plus_b的第一个维度（axes=[0]，对于本例数据而言第一维即batch）的均值和方差
	            scale = tf.Variable(tf.ones([out_size]))
	            shift = tf.Variable(tf.zeros([out_size]))
	            epsilon = 0.001
	            #这些参数都是供BN所使用的

	            ema = tf.train.ExponentialMovingAverage(decay=0.5)
	            #定义衰减率decay=0.5的滑动平均的方法。
	            #因为使用 batch 进行每次的更新, 那每个 batch 的 mean/var 都会不同, 所以我们可以使用 moving average 的方法记录并慢慢改进 mean/var 的值. 
	            #然后将修改提升后的 mean/var 放入 tf.nn.batch_normalization()。
	            def mean_var_with_update():
	                ema_apply_op = ema.apply([fc_mean, fc_var])
	                #将均值和方差用滑动平均方法更新
	                with tf.control_dependencies([ema_apply_op]):
	                #定义运算控制器tf.control_dependencies，为了控制调用顺序，只有执行了ema_apply_op才能执行以下语句
	                    return tf.identity(fc_mean), tf.identity(fc_var)
	                    #tf.identity函数指不对参数修改，直接返回原值。这里使用的意图是？
	            mean, var = mean_var_with_update()   
	            # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var

	            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
	            # 相当于进行了以下两步计算，即BN的归一化
	            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
	            # Wx_plus_b = Wx_plus_b * scale + shift

	        if activation_function is None:
	            outputs = Wx_plus_b
	        else:
	            outputs = activation_function(Wx_plus_b)

	        return outputs

	    fix_seed(1)
	    #调用fix_seed

	    if norm:
	        # BN for the first input
	        fc_mean, fc_var = tf.nn.moments(xs,axes=[0],)
	        scale = tf.Variable(tf.ones([1]))
	        shift = tf.Variable(tf.zeros([1]))
	        epsilon = 0.001
	        # apply moving average for mean and var when train on batch
	        ema = tf.train.ExponentialMovingAverage(decay=0.5)
	        def mean_var_with_update():
	            ema_apply_op = ema.apply([fc_mean, fc_var])
	            with tf.control_dependencies([ema_apply_op]):
	                return tf.identity(fc_mean), tf.identity(fc_var)
	        mean, var = mean_var_with_update()
	        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)

	    layers_inputs = [xs]
	    #用layers_inputs记录输入的数据

	    # build hidden layers
	    for l_n in range(N_LAYERS):
	        layer_input = layers_inputs[l_n]
	        in_size = layers_inputs[l_n].get_shape()[1].value

	        output = add_layer(layer_input,in_size,N_HIDDEN_UNITS, ACTIVATION,norm,)
	        #这时才调用add_layer函数
	        layers_inputs.append(output)    # add output for next run

	    # build output layer
	    prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)

	    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
	    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
	    return [train_op, cost, layers_inputs]

	# 整个函数从这里开始
	fix_seed(1)
	#调用开头定义的fix_seed函数
	x_data = np.linspace(-7, 10, 2500)[:, np.newaxis]
	np.random.shuffle(x_data)
	noise = np.random.normal(0, 8, x_data.shape)
	y_data = np.square(x_data) - 5 + noise

	plt.scatter(x_data, y_data)
	plt.show()
	#画x和y的点状图

	xs = tf.placeholder(tf.float32, [None, 1])  # [num_samples, num_features]
	ys = tf.placeholder(tf.float32, [None, 1])

	train_op, cost, layers_inputs = built_net(xs, ys, norm=False)  
	#建立一个没有使用BN的网络，这里才开始调用built_net函数。但是还没运行，输入值xs, ys还没给。需要后期run
	train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True) 
	#建立一个使用BN的网络

	sess = tf.Session()
	if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	    init = tf.initialize_all_variables()
	else:
	    init = tf.global_variables_initializer()
	sess.run(init)

	# record cost
	cost_his = []
	cost_his_norm = []
	record_step = 5

	plt.ion()
	plt.figure(figsize=(7, 3))
	for i in range(250):
	    if i % 50 == 0:
	        # 每50次画一下直方图
	        all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data})
	        #这时候run，想得到all_inputs, all_inputs_norm，故跳去1212和1214，将x_data, y_data赋值给xs，ys，这时built_net才开始运行，跳去1120行
	        plot_his(all_inputs, all_inputs_norm)

	    # train on batch
	    sess.run([train_op, train_op_norm], feed_dict={xs: x_data[i*10:i*10+10], ys: y_data[i*10:i*10+10]})

	    if i % record_step == 0:
	        # 每5次记录一下cost
	        cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))
	        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))

	plt.ioff()
	plt.figure()
	plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his), label='no BN')     # no norm
	plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his_norm), label='BN')   # norm
	plt.legend()
	plt.show(

#
实例：Q学习简单例子 —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	import numpy as np
	import pandas as pd
	import time

	np.random.seed(2)  # reproducible


	N_STATES = 6   # 1维世界的宽度
	ACTIONS = ['left', 'right']     # 探索者的可用动作
	EPSILON = 0.9    # 贪婪度 greedy
	ALPHA = 0.1     # 学习率
	GAMMA = 0.9   # 奖励递减值
	MAX_EPISODES = 13     # 最大回合数
	FRESH_TIME = 0.3     # 移动间隔时间

	#定义Q表
	def build_q_table(n_states, actions):
	    table = pd.DataFrame(
	        np.zeros((n_states, len(actions))),     # q_table 全 0 初始
	        columns=actions,     # columns 对应的是行为名称
	    )
	    # print(table)    # show table
	    return table

	# 在某个 state 地点, 选择行为
	def choose_action(state, q_table): 
	    state_actions = q_table.iloc[state, :]
	    # DataFrame.iloc函数，纯粹的基于整数位置的索引，用于按位置进行选择
	    # 选出这个 state 行的所有 action 列的值
	    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  
	    # EPSILON是用来控制贪婪程度的值，EPSILON = 0.9, 意味着90% 的时间是选择最优策略, 10% 的时间来探索
	    # 因为np.random.uniform()产生的是0-1之间任意值，故可用0.9的来代表控制策略选择的阈值
	    # 非贪婪 or 或者这个 state 还没有探索过
	        action_name = np.random.choice(ACTIONS)
	        #np.random.choice随机选择。即随机探索选择向左或向右的动作。
	    else:   # 贪婪模式
	        action_name = state_actions.idxmax()    # 挑选对应值最大的action
	    return action_name


	def get_env_feedback(S, A):
	    # 定义一个奖励函数
	    if A == 'right':    # 若向右走
	        if S == N_STATES - 2:   # 若当前状态为终点（位置为总长-1，即6-1=5）前一步，6-2=4
	            S_ = 'terminal'#则下一个状态为终点
	            R = 1#奖励设为1
	        else:
	            S_ = S + 1 #否则下一状态为s+1
	            R = 0#奖励0
	    else:   # 因为终点设在最右端故向左走都是错的
	        R = 0 #奖励为0 
	        if S == 0:
	            S_ = S  # 碰壁了
	        else:
	            S_ = S - 1
	    return S_, R


	def update_env(S, episode, step_counter):
	    # This is how environment be updated
	    env_list = ['-']*(N_STATES-1) + ['T']  
	    #N_STATES-1个‘-’和一个‘T’，所以整个环境是 '---------T' 
	    if S == 'terminal':
	    #若当先状态s是'terminal'则输出第n次迭代总步数
	        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
	        print('\r{}'.format(interaction), end='')
	        time.sleep(2)
	        print('\r                                ', end='')
	    else:
	    #若当前不是终点
	        env_list[S] = 'o'
	        #把当前位置s对应的env_list里的值设为o
	        interaction = ''.join(env_list)
	        #.join是联合函数，例如 a="abcd"； ",".join(a)-----'a,b,c,d'
	        print('\r{}'.format(interaction), end='')
	        time.sleep(FRESH_TIME)


	def rl():
	    # 强化学习的主循环部分
	    q_table = build_q_table(N_STATES, ACTIONS)
	    for episode in range(MAX_EPISODES):
	        step_counter = 0
	        S = 0
	        is_terminated = False
	        update_env(S, episode, step_counter)
	        while not is_terminated:

	            A = choose_action(S, q_table)
	            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
	            q_predict = q_table.ix[S, A]
	            #.ix函数也用作按序号索引选取table中的元素。从Q表中找到s状态下采取a行动的值
	            #q_predict是当前s选a的评估
	            if S_ != 'terminal':
	                q_target = R + GAMMA * q_table.iloc[S_, :].max()  
	                # q_target是取的当前奖励R和下一状态S_所有行为a的最大值
	                # 若不是终点，则R肯定为0,实际的(状态-行为)值 (回合没结束)
	            else:
	                q_target = R      #  实际的(状态-行为)值 (回合结束)
	                is_terminated = True    # terminate this episode

	            q_table.ix[S, A] += ALPHA * (q_target - q_predict)   #  q_table 用两者的差值更新
	            S = S_  # move to next state

	            update_env(S, episode, step_counter+1)
	            step_counter += 1
	    return q_table


	if __name__ == "__main__":
	    q_table = rl()
	    print('\r\nQ-table:\n')
	    print(q_table

#
实例：sarsa和Ql对比—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	"""
	This part of code is the Q learning brain, which is a brain of the agent.
	All decisions are made in here.
	View more on my tutorial page: https://morvanzhou.github.io/tutorials/
	"""

	import numpy as np
	import pandas as pd

	class RL(object):
	#定义一个包含三种RL方法（QLearningTable、SarsaTable和SarsaLambdaTable）的类
	#RL类的所有对象都包含__init__，check_state_exist，choose_action和learn方法
	    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
	        self.actions = action_space  # a list
	        self.lr = learning_rate
	        self.gamma = reward_decay
	        self.epsilon = e_greedy

	        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

	    def check_state_exist(self, state):
	    #检查目前状态state是否存在函数
	        if state not in self.q_table.index:
	            # 如果state不在q_table.index内
	            self.q_table = self.q_table.append(
	                pd.Series(
	                    [0]*len(self.actions),
	                    index=self.q_table.columns,
	                    name=state,
	                )
	            )

	    def choose_action(self, observation):
	        self.check_state_exist(observation)
	        # action selection
	        if np.random.rand() < self.epsilon:
	            # choose best action
	            state_action = self.q_table.ix[observation, :]
	            state_action = state_action.reindex(np.random.permutation(state_action.index))     
	            # 如果用一个s下，有多个a对应同样的Q值，会导致在选的时候总是选到第一个这样的a，而位于后面的则没机会被选到，所以需要随机打乱一下排序
	            action = state_action.argmax()
	        else:
	            # choose random action
	            action = np.random.choice(self.actions)
	        return action

	    def learn(self, *args):
	        pass
	        # 每种的都有点不同, 所以用 pass


	# off-policy
	#q学习作为sarsa的改进，区别就在于使用了off-policy，即更新Q值（q_target）时使用的是s_状态下所有a对应的最大得分。
	class QLearningTable(RL):
	#定义一个归属于父类RL的子类QLearningTable
	    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
	        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
	        #super用作调用父类方法

	    def learn(self, s, a, r, s_):
	    #传入参数包含当前状态s，当前动作a，累计分数r和下一状态s_
	        self.check_state_exist(s_)
	        q_predict = self.q_table.ix[s, a]
	        if s_ != 'terminal':
	            q_target = r + self.gamma * self.q_table.ix[s_, :].max() 
	            # self.q_table.ix[s_, :].max() 即不使用固定的a，而是找最大的a所对应的值。
	        else:
	            q_target = r  # next state is terminal
	        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update


	# on-policy
	#重点区分q_target
	class SarsaTable(RL):
	    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
	        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

	    def learn(self, s, a, r, s_, a_):
	    #这里的传入参数多了一个下一步动作a_
	        self.check_state_exist(s _)
	        q_predict = self.q_table.ix[s, a]
	     ko    if s_ != 'terminal':
	            q_target = r + self.gamma * self.q_table.ix[s_, a_] 
	            # 乘的是在s_状态下采取a_动作所对应的奖励值
	        else:
	            q_target = r  # next state is terminal
	        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

	#进化版Sarsa，引入了一个参数lambda，控制最能影响结果的Q（s，a）的重要性
	class SarsaLambdaTable(RL):
	    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
	        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

	        # 除了继承父类的init以外还需要两个参数
	        self.lambda_ = trace_decay
	        #是（0,1）之间的数，衰减值。
	        self.eligibility_trace = self.q_table.copy()
	        #类似于Q表，用于记录（s，a），每次加1

	    def check_state_exist(self, state):
	        if state not in self.q_table.index:
	            # append new state to q table
	            to_be_append = pd.Series(
	                    [0] * len(self.actions),
	                    index=self.q_table.columns,
	                    name=state,
	                )
	            self.q_table = self.q_table.append(to_be_append)

	            # 除了更新Q表，还需要更新eligibility_trace
	            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

	    def learn(self, s, a, r, s_, a_):
	        self.check_state_exist(s_)
	        q_predict = self.q_table.ix[s, a]
	        if s_ != 'terminal':
	            q_target = r + self.gamma * self.q_table.ix[s_, a_]  # next state is not terminal
	        else:
	            q_target = r  # next state is terminal
	        error = q_target - q_predict

	        # 这里开始不同:
	        # Method 1:
	        # 对于经历过的 state-action, 我们让他+1, 证明他是得到 reward 路途中不可或缺的一环
	        # self.eligibility_trace.ix[s, a] += 1
	        # 但这种方法每次加1可能会造成有些（s，a）太大。

	        # Method 2:
	        self.eligibility_trace.ix[s, :] *= 0
	        self.eligibility_trace.ix[s, a] = 1
	        #让eligibility_trace表除了（s，a）以外的值全部为0，而其本身设为1

	        # Q table 更新，用到了eligibility_trace
	        self.q_table += self.lr * error * self.eligibility_trace

	        # 随着时间衰减 eligibility trace 的值, 离获取 reward 越远的步, 他的重要性越小
	        self.eligibility_trace *= self.gamma*self.lambda_

#
实例：DQN————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
	"""
	This part of code is the Deep Q Network (DQN) brain.

	view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

	View more on my tutorial page: https://morvanzhou.github.io/tutorials/

	Using:
	Tensorflow: r1.2
	"""

	import numpy as np
	import tensorflow as tf

	np.random.seed(1)
	tf.set_random_seed(1)


	# Deep Q Network off-policy
	class DeepQNetwork:
	    def __init__(
	            self,
	            n_actions,
	            n_features,
	            learning_rate=0.01,
	            reward_decay=0.9,
	            e_greedy=0.9,
	            replace_target_iter=300,
	            memory_size=500,
	            batch_size=32,
	            e_greedy_increment=None,
	            output_graph=False,
	    ):
	        self.n_actions = n_actions
	        self.n_features = n_features
	        self.lr = learning_rate
	        self.gamma = reward_decay
	        self.epsilon_max = e_greedy
	        self.replace_target_iter = replace_target_iter
	        self.memory_size = memory_size
	        self.batch_size = batch_size
	        self.epsilon_increment = e_greedy_increment
	        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

	        # total learning step
	        self.learn_step_counter = 0

	        # initialize zero memory [s, a, r, s_]
	        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

	        # consist of [target_net, evaluate_net]
	        self._build_net()

	        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
	        #tf.get_collection是从一个结合中取出全部变量。即整合名为target_net的网络中所有变量
	        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

	        with tf.variable_scope('soft_replacement'):
	            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
	            #tf.assign将t变为e，t、e分别来自t_params, e_params

	        self.sess = tf.Session()

	        if output_graph:
	            # $ tensorboard --logdir=logs
	            tf.summary.FileWriter("logs/", self.sess.graph)

	        self.sess.run(tf.global_variables_initializer())
	        self.cost_his = []

	    def _build_net(self):
	        # ------------------ all inputs ------------------------
	        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
	        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
	        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
	        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

	        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

	        #-------------- 创建 eval 神经网络, 及时提升参数 --------------
	        #两层全连接层，第一层激活函数relu，输入同变量s个数，输出20个，名为e1。第二层输入同e个数，输出同行动n_actions个数，名为q。
	        with tf.variable_scope('eval_net'):
	            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
	                                 bias_initializer=b_initializer, name='e1')
	            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
	                                          bias_initializer=b_initializer, name='q')

	        # ---------------- 创建 target 神经网络, 提供 target Q ---------------------
	        #与eval网络结构完全相同，第一层输出名为t1，第二层输出为t2.
	        with tf.variable_scope('target_net'):
	            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
	                                 bias_initializer=b_initializer, name='t1')
	            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
	                                          bias_initializer=b_initializer, name='t2')

	        with tf.variable_scope('q_target'):
	            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    
	            # 计算目标Q值，r是先前rewards，加target网络输出Q值的行（axis=1）最大值（reduce_max）乘以衰减系数gamma
	            self.q_target = tf.stop_gradient(q_target)
	            #tf.stop_gradient使得q_target这个节点的梯度值停止。从而完成停止bp进程，不再随梯度更行的目的。

	        with tf.variable_scope('q_eval'):
	            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
	            #通过tf.shape(self.a)获取a的尺寸，用range生成长度为a尺寸类行为int32的序列，用tf.stack将前者和self.a按行（axis=1）合并成一个序号和行动对应表
	            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)   
	            #用a_indices从张量self.q_eval得到新张量，意义是？？？？？？？？？？？？？？？？？？？？？？
	        with tf.variable_scope('loss'):
	            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
	            #使用q_target和平方差均值作为loss
	        with tf.variable_scope('train'):
	            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

	    def store_transition(self, s, a, r, s_):
	        if not hasattr(self, 'memory_counter'):
	            self.memory_counter = 0
	        #更新记忆更迭计数器
	        transition = np.hstack((s, [a, r], s_))
	        # 记录一条 [s, a, r, s_] 记录
	        index = self.memory_counter % self.memory_size
	        #索引为当前记忆量除以规定记忆量标准，总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
	        self.memory[index, :] = transition
	        #替换过程。例如size=5，则counter是0-4时index都0，这时传入的所有transition均被放在0行的那几个位置，逐次替换。直到counter=5才换为第一行
	        self.memory_counter += 1

	    def choose_action(self, observation):
	        #统一 observation 的 shape (1, size_of_observation)
	        observation = observation[np.newaxis, :]

	        if np.random.uniform() < self.epsilon:
	            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
	            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
	            action = np.argmax(actions_value)
	        else:
	            action = np.random.randint(0, self.n_actions)
	        return action

	    def learn(self):
	        # 检查是否替换 target_net 参数
	        if self.learn_step_counter % self.replace_target_iter == 0:
	            self.sess.run(self.target_replace_op)
	            print('\ntarget_params_replaced\n')

	        # 从 memory 中随机抽取 batch_size 这么多记忆
	        if self.memory_counter > self.memory_size:
	            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
	        else:
	            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
	        batch_memory = self.memory[sample_index, :]

	        _, cost = self.sess.run(
	            [self._train_op, self.loss],
	            feed_dict={
	                self.s: batch_memory[:, :self.n_features],
	                self.a: batch_memory[:, self.n_features],
	                self.r: batch_memory[:, self.n_features + 1],
	                self.s_: batch_memory[:, -self.n_features:],
	            })

	        self.cost_his.append(cost)

	        # increasing epsilon
	        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
	        self.learn_step_counter += 1

	    def plot_cost(self):
	        import matplotlib.pyplot as plt
	        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
	        plt.ylabel('Cost')
	        plt.xlabel('training steps')
	        plt.show()

	if __name__ == '__main__':
	    DQN = DeepQNetwork(3,4, output_graph=True)
