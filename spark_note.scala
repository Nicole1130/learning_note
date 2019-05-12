/*wordCounts
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
*/
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

val conf  = new SparkConf  //创建spark配置对象。可以简写成val conf = new SparkConf().setMaster("spark://server1:8888").setAppName("my first spark") 
conf.setAppName("my first spark")  //设置在程序运行的监控界面可以看到的名称
conf.setMaster("local")  //设置在本地运行。"local[4]" 使用本地的4核cpu。或者填"spark://master:7077" 在spark集群上运行，主机名称(或主机ip)：端口号.

val sc = new SparkContext(conf) //读取配置并初始化spark的功能核心组件context，它是所有功能的唯一入口，必有。

val lines = sc.textFile("D://README.md",1) //初始化一个rdd名叫lines，用于读取path文件，分区为Partition(1,也可不设定)。path是一个String类型URI，可以是HDFS、本地文件
val words = lines.flatMap{line => line.split(" ")} //读取来的文本以行写入，遍历时每次的对象为一行line，对这一行进行空格拆分变为单词集合words。
                 								   //因为每一行是一个集合，故需要用flatmap将每行分出来的集合words压平合并flat成一个大集合.
val pairs = words.map{word => (word,1)} // 遍历出每一个word并将其变成(word,1)元祖
val wordCounts = pairs.reduceByKey(_ + _) //reduceByKey会找相同key的数据，当找到这样的两条记录时会对其value(分别记为x,y)做(x,y) => x+y的处理(简写为_+_)，如此反复
wordCounts.foreach(pairs => println(pairs._1+":"+pairs._2)) //对于wordCounts中的每组pairs(此处的pairs和前面定义的val不是同一个，只是一个遍历时用的局部变量，用后销毁)
															//输出foreach的遍历与map遍历区别在于前者不对变量进行修改，只简单操作，例如此处的执行输出操作。
sc.stop() //关闭spark 



/*MLlib基本数据类型
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
*/
import org.apache.spark.mllib.linalg.{Vector, Vectors}
val dv: Vector = Vectors.dense(2.0, 0.0, 8.0) // 稠密向量,使用一个双精度浮点型数组来表示其中每一维元素,表示形式是[1.0,0.0,8.0]
val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(2.0, 8.0)) //稀疏向量,则是基于一个整型索引数组和一个双精度浮点型的值数组(3, [0,2], [1.0, 3.0])，
																  //其中，3是向量的长度，[0,2]是向量中非0维度的索引值，表示位置为0、2的两个元素为非零值，
																  //而[2.0, 8.0]则是按索引排列的数组元素值
import org.apache.spark.mllib.regression.LabeledPoint
val pos = LabeledPoint(1.0, Vectors.dense(2.0, 0.0, 8.0)) //标注点LabeledPoint是一种带有标签（Label/Response）的本地向量，它可以是稠密或者是稀疏的.
														  //表示创建一个标签为1.0（分类中可视为正样本）的稠密向量标注点
import org.apache.spark.mllib.util.MLUtils
val examples = MLUtils.loadLibSVMFile(sc, "/data/mllib/sample_libsvm_data.txt") //用loadLibSVMFile方法读入LIBSVM格式数据,返回一系列LabeledPoint的rdd
examples.collect().head //examples.collect()把rdd转换为了向量，并取第一个元素的值。返回org.apache.spark.mllib.regression.LabeledPoint = (0.0,(5,[2,3],[5，2]))
						//意为，第一个元素类别为0(负样本)，共有5个维，其中第2、3列对应的值分别是5，2。
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0)) //创建一个3行2列的稠密矩阵[[1.0,2.0], [3.0,4.0], [5.0,6.0]],数组参数是列先序的！
val sm: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8)) //创建一个3行2列的稀疏矩阵[ [9.0,0.0], [0.0,8.0], [0.0,6.0]]
																					   //第一个Array是列序号，第2个是行序号。例如0行0列存放9,2行1列放6。。。

val numbers = 1 to 20 //生成一个1到20的序列
val rdd = sc.parallelize(numbers)  //使用.parallelize或者.makeRDD将程序中的集合转为rdd。可以指定分区数,默认为cpu核数。
val rdd = sc.textFile()

val dv1 : Vector = Vectors.dense(1.0,2.0,3.0)
val dv2 : Vector = Vectors.dense(2.0,3.0,4.0)
val rows : RDD[Vector] = sc.parallelize(Array(dv1,dv2)) //使用两个本地向量创建一个RDD[Vector]
val mat : RowMatrix = new RowMatrix(rows) //通过RDD[Vector]创建一个行矩阵
mat.numRows() //得到行数 res0: Long = 2
mat.numCols() //得到列数 res1: Long = 3
mat.rows.foreach(println) //每个元素执行println输出函数，得到[1.0,2.0,3.0][2.0,3.0,4.0]
val summary = mat.computeColumnSummaryStatistics() //通过computeColumnSummaryStatistics()方法获取统计摘要
summary.count //行数,2 
summary.max  //最大向量 [2.0,3.0,4.0]
summary.variance //方差向量 [0.5,0.5,0.5]
summary.mean //平均向量 [1.5,2.5,3.5]
summary.normL1 //L1范数向量 [3.0,5.0,7.0]

val idxr1 = IndexedRow(1,dv1) //给dv1的向量赋索引值1,生成IndexedRow(1,[1.0,2.0,3.0])
val idxr2 = IndexedRow(2,dv2)  
val idxrows = sc.parallelize(Array(idxr1,idxr2)) //通过IndexedRow创建RDD[IndexedRow]
val idxmat: IndexedRowMatrix = new IndexedRowMatrix(idxrows) //通过RDD[IndexedRow]创建一个索引行矩阵

val ent1 = new MatrixEntry(0,1,0.5) //创建两个矩阵项ent1和ent2，每一个矩阵项都是由索引和值构成的三元组,其中0是行索引，1是列索引，0.5是该位置的值
val ent2 = new MatrixEntry(2,2,1.8)
val entries : RDD[MatrixEntry] = sc.parallelize(Array(ent1,ent2)) //创建RDD[MatrixEntry]
val coordMat: CoordinateMatrix = new CoordinateMatrix(entries) //通过RDD[MatrixEntry]创建一个坐标矩阵
val transMat: CoordinateMatrix = coordMat.transpose() //将coordMat进行转置.变成  MatrixEntry(1,0,0.5)，MatrixEntry(2,2,1.8)
val indexedRowMatrix = transMat.toIndexedRowMatrix()  // 将坐标矩阵转换成一个索引行矩阵

/*摘要统计 Summary statistics
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
*/
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
val observations=sc.textFile("G:/spark/iris.data").map(_.split(",")).map(             //读入iris数据集，其一行的格式为5.1,3.5,1.4,0.2,Iris-setosa
	p => Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble))   //数据转变成RDD[Vector]类型
val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)  //调用colStats()方法，得到MultivariateStatisticalSummary类型的变量
summary.count //列的大小
summary.mean //每列的均值
summary.variance //每列的方差
summary.max //每列的最大值
summary.normL1 //每列的L1范数
summary.numNonzeros //每列非零向量的个数

val seriesX = sc.textFile("G:/spark/iris.data").map(_.split(",")).map(p => p(0).toDouble) 
val seriesY = sc.textFile("G:/spark/iris.data").map(_.split(",")).map(p => p(1).toDouble) 
val correlation: Double = Statistics.corr(seriesX, seriesY, "pearson") //调用Statistics包中的corr()函数来获取相关性
val data = sc.textFile("G:/spark/iris.data").map(_.split(",")).map(p => Vectors.dense(p(0).toDouble, p(1).toDouble))
val correlMatrix1: Matrix = Statistics.corr(data, "pearson") //或者直接调用矩阵求相关系数矩阵，结果是一致

import org.apache.spark.SparkContext
import org.apache.spark.mllib.random.RandomRDDs._ //随机数生成 
val u = normalRDD(sc, 10000000L, 10) // 生成1000000个服从正态分配N(0,1)的RDD[Double]，并且分布在 10 个分区中.
val v = u.map(x => 1.0 + 2.0 * x) // 把生成的随机数转化成N(1,4) 正态分布.

import org.apache.spark.mllib.stat.KernelDensity
import org.apache.spark.rdd.RDD  // 核密度估计 Kernel density estimation
val test = sc.textFile("G:/spark/iris.data").map(_.split(",")).map(p => p(0).toDouble)//用假设检验中得到的iris的第一个属性的数据作为样本数据进行估计
val kd = new KernelDensity().setSample(test).setBandwidth(3.0) //用样本数据构建核函数，setBandwidth表示高斯核的宽度，为一个平滑参数，可看做高斯核的标准差
val densities = kd.estimate(Array(-1.0, 2.0, 5.0, 5.8))//返回Array[Double] = Array(0.011, 0.059, 0.12, 0.12)，表示在四个样本点上估算的概率密度函数值


/* 降维
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
*/

import org.apache.spark.mllib.linalg.distributed.RowMatrix // 奇异值分解SVD
val data = sc.textFile("a.mat").map(_.split(" ").map(_.toDouble)).map(line => Vectors.dense(line)) //a中存放一个(4,9)的矩阵，每个元素用“ ”间隔
val rm = new RowMatrix(data) //通过RDD[Vectors]创建行矩阵
val svd = rm.computeSVD(3) //保留前3个奇异值
svd.s //因为限定了取前三个奇异值，所以奇异值向量s包含有三个从大到小排列的奇异值
svd.V //右奇异矩阵V中的每一列都代表了对应的右奇异向量。
svd.U //U成员得到的是一个null值，这是因为在实际运用中，只需要V和S两个成员，即可通过矩阵计算达到降维的效果.如需则rm.computeSVD(3, computeU = true)设为真即可


import org.apache.spark.mllib.linalg.Vectors  //主成分分析PCA
import org.apache.spark.mllib.linalg.distributed.RowMatrix
val data = sc.textFile("a.mat").map(_.split(" ").map(_.toDouble)).map(line => Vectors.dense(line))
val rm = new RowMatrix(data)
val pc = rm.computePrincipalComponents(3) //将a.mat（4*9）矩阵看成是一个有4个样本，9个特征的数据集，执行输出为(9*3)的矩阵,
										  //每列代表一个主成分（新坐标轴），每行代表原有的一个特征.相当于把原有的9维特征空间投影到一个3维的空间
val projected = rm.multiply(pc) //通过矩阵乘法来完成对原矩阵的PCA变换，可以看到原有的(4,9)矩阵被变换成新的(4,3)矩阵

import org.apache.spark.mllib.feature.PCA //“模型式”的PCA变换实现.适用于原始数据是LabeledPoint类型的情况，只取LabeledPoint的feature成员（RDD[Vector]类型）
import org.apache.spark.mllib.regression.LabeledPoint //，对其做PCA操作后再放回，即可在不影响原有标签情况下进行PCA变换
val data = sc.textFile("a.mat").map(_.split(" ").map(_.toDouble)).map(line =>              //用前文的a.mat矩阵，创造一个 LabeledPoint
	{ LabeledPoint( if(line(0) > 1.0) 1.toDouble else 0.toDouble, Vectors.dense(line) )})  //第一个样本(第一个字符 line(0) < 1.0))标注标签为0.0，其他为1.0
val pca = new PCA(3).fit(data.map(_.features)) //创建一个PCA类对象，构造器中定主成分数3，调用其fit方法来生成一个PCAModel类的对象pca，该对象保存了对应的主成分矩阵
val projected = data.map(p => p.copy(features = pca.transform(p.features))) //LabeledPoint型的数据可用map算子对每一条数据进行处理，将features替换成PCA变换后的特征

/* 逻辑回归
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
*/

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors,Vector}
val data = sc.textFile("G:/spark/iris.data") //数据集每行被分成了5部分，前4部分是鸢尾花的4个特征，最后一部分是鸢尾花的分类
val parsedData = data.map { line => //LabeledPoint要求标签的类型是double，特征的类型是Vector."Iris-setosa"对应分类0,以此类推
     |     val parts = line.split(',')
     |     LabeledPoint(if(parts(4)=="Iris-setosa") 0.toDouble else if (parts(4)=="Iris-versicolor") 1.toDouble else 2.toDouble, 
     |	   Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts(2).toDouble,parts(3).toDouble)) }
val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L) //划分60%的训练集和40%的测试集
val training = splits(0).cache() //.cache先放入缓存以待后续使用。
val test = splits(1)
val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(training) //构建逻辑回归模型，用set设置分3类，在training上训练
val predictionAndLabels = test.map { case LabeledPoint(label, features) => //将test中每一个元素映射成LabeledPoint(label, features)
     |       val prediction = model.predict(features) //模型输入特征得到预测结果
     |       (prediction, label)} //把预测值和真正的标签放到predictionAndLabels     
val metrics = new MulticlassMetrics(predictionAndLabels) //将预测结果和真实结果放入多分类矩阵MulticlassMetrics，用于后续模型评估
val precision = metrics.precision //得到模型预测的准确性。输出 precision: Double = 0.9180327868852459



/* 决策树
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
*/

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors,Vector}
val data = sc.textFile("G:/spark/iris.data") //依然使用iris数据集
val parsedData = data.map { line => 
     |     val parts = line.split(',')
     |     LabeledPoint(if(parts(4)=="Iris-setosa") 0.toDouble else if (parts(4)=="Iris-versicolor") 1.toDouble else 2.toDouble, 
     |	   Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts(2).toDouble,parts(3).toDouble)) }
val splits = parsedData.randomSplit(Array(0.7, 0.3))  //随机7：3划分数据集
val (trainingData, testData) = (splits(0), splits(1)) 
val numClasses = 3  //分类个数
val categoricalFeaturesInfo = Map[Int, Int]() //categoricalFeaturesInfo 为空，意味着所有的特征为连续型变量
val impurity = "gini" //纯度计算
val maxDepth = 5 //树的最大层次
val maxBins = 32 //特征最大装箱数
val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins) //建立决策树
val labelAndPreds = testData.map { point => val prediction = model.predict(point.features)  //使用决策树预测
     |       (point.label, prediction) } //返回预测值和真正的标签
println(model.toDebugString) //显示模型结构。例如 DecisionTreeModel classifier of depth 5 with 15 nodes
								//  If (feature 2 <= 1.9)
								//      Predict: 0.0
								//  Else (feature 2 > 1.9)
								//      If (feature 2 <= 4.8)
			                    //      。。。
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count() //评估模型，计算错误率


/* 支持向量机SVM
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
*/
import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.classification.SVMWithSGD
val data = sc.textFile("G:/spark/iris.data") //依然使用iris数据集
val parsedData = data.map { line => 
     |     val parts = line.split(',')
     |     LabeledPoint(if(parts(4)=="Iris-setosa") 0.toDouble else if (parts(4)=="Iris-versicolor") 1.toDouble else 2.toDouble, 
     |	   Vectors.dense(parts(0).toDouble,parts(1).toDouble,parts(2).toDouble,parts(3).toDouble)) }
val splits = parsedData.randomSplit(Array(0.7, 0.3))  //随机7：3划分数据集
val (trainingData, testData) = (splits(0), splits(1)) 
val numIterations = 100 //迭代次数
val model = SVMWithSGD.train(trainingData, numIterations)// 建立svm模型并训练
val predictionAndLabel = testData.map { point => val score = model.predict(point.features)  // 对测试样本进行测试
   |                     (score, point.label, point.features)}
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testRDD.count()// 评估模型，计算准确率



/* 协同过滤算法
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
*/
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS //ALS交替最小二乘法（Alternating Least Squares）,是一个基于模型的协同过滤
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
val data = sc.textFile("../data/mllib/als/test.data") //文件中每一行包括一个用户id、商品id和评分
val ratings = data.map(_.split(',') match { case Array(user, item, rate) 
                      |	  => Rating(user.toInt, item.toInt, rate.toDouble)}) //把数据转化成rating类型，即[Int, Int, Double]的RDD
val splits = ratings.randomSplit(Array(0.8, 0.2))
val training = splits(0)
val test = splits(1)
val rank = 10  //隐藏因子的个数
val numIterations = 10  //迭代次数
val model = ALS.train(training, rank, numIterations, 0.01) //使用ALS训练数据建立推荐模型.0.01是ALS的正则化参数
val testUsersProducts = test.map { case Rating(user, product, rate) => (user, product)} //从test训练集中获得只包含用户和商品的数据集
val predictions = model.predict(testUsersProducts).map { case Rating(user, product, rate)=>((user, product), rate) } //对用户商品预测评分
val ratesAndPreds = test.map { case Rating(user, product, rate) =>
                    |  ((user, product), rate) }.join(predictions)//join对两者都存在的内容操作，其余被过滤。整理后变为((用户，产品)，真实评分，预测分)
val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>val err = (r1 - r2) err * err}.mean() //map返回err*err，再对其求均值，得出均方误差





/* KMean聚类
-----------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------
*/
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
val rawData = sc.textFile("iris.csv") //每行数据格式例如，5.1,3.5,1.4,0.2,setosa
val trainingData = rawData.map(line => {Vectors.dense(line.split(",").filter(     //每行以逗号分割元素，过滤出是数字的元素，转为double形式并放入缓存
	            | p => p.matches("\\d*(\\.?)\\d*")).map(_.toDouble))}).cache() //“\\d”表示0-9的数字，“*”是限定符表示匹配0次或多次，“\\.?”使用了?限定符，表示匹配0次或1次的小数点
val model : KMeansModel = KMeans.train(trainingData, 3, 100, 5) // 构建kmean模型，聚类数目3，最大迭代100次，运行5次
model.clusterCenters.foreach(center => {println("Clustering Center:"+center)}) //输出3个中心点的数据，例如中心点1是[6.3,2.9,5.0,1.7]，因为输入数据有4维特征所以中心点也是4维坐标
trainingData.collect().foreach(sample => {val predictedCluster = model.predict(sample) //对于每一个训练集中样本进行预测并输出结果
     |   println(sample.toString + " belongs to cluster " + predictedCluster)})   //collect操作的特点是从远程集群是拉取数据到本地,foreach是在远程集群上遍历rdd中的元素
val wssse = model.computeCost(trainingData) //计算集合内误差平方和,度量聚类的有效性.对于那些无法预先知道K值的情况,可以通过WSSSE构建出K-WSSSE间的相关关系，从而确定K的值
