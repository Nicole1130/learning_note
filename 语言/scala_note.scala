scala > val helloString : String = "HelloVal" //val声明不可变变量,': String'变量声明可写可不写，系统会自动判断
helloString : String = HelloVal  //输出变量名、类型和结果
scala > var helloString  = "HelloVar" //var声明可变变量
helloString : String = HelloVar  //输出变量名、类型和结果
scala > var helloString:String  = "" //var和val变量都必须初始化赋值
scala > var helloString:Int  = _ //_是占位符，可以对任何类型变量使用
scala > val helloString : String = " \"HelloVal\" " // \"是转义字符，表示“
helloString : String = "HelloVal"  //输出带有引号
scala > println("""helo ,\\ \n """) //使用"""能原样输出内容
helo ,\\ \n  //输出原样字符串内容


//符号，常规用法就不再说明
def 函数名 (输入名：类型 ) =（返回名：类型）表达式  //如果有返回值的运算，例如 = x match...需要加=号
def 函数名 (输入名：类型 ) { 表达式 }              //两种定义法同效果，没有对输出的特殊运算不必要加=号（加也没关系），直接用{}代替，只有一句表达式{}也可以省略

Map(变量1 -> 变量2)     //->用于map或flatMap映射
case 变量名: 类型 => 表达式    //=>号似乎只能用再模式匹配时
for(临时变量 <- 集合)    //<- 用于for遍历



scala > 1 + 2L //不同类型可以直接操作，可以直接运行表达式
res0 : Long = 3  //结果自动转类别，且被默认命名为res,可以在后续操作中继续使用res这个名称
scala > 1 + -3 //可以直接使用+-代表正负
res1 : Int = -2  


scala > X = "hello" //不同类型可以直接操作，可以直接运行表达式
scala > Y = "hello"  //可以直接使用+-代表正负
scala > Y == X //scala的==和java不一样，还可以用y.equals(x),前者比较内容相等，后者比较引用相等（内存地址）。返回True
scala > Y.eq(x) //比较内存地址，返回false

//元祖
scala > ("a","b",1) // 定义元祖，可以存储不同类型值
res0 : (String,String,Int) = ("a","b",1)
scala > res0._2  //元祖可以通过_n的方式索引,而且是从1开始，不是从0开始
res1 : string "b"
"New Year".partition(_.isUpper) //partition返回一对元祖，满足条件（.isUpper是大写字符）的存在_1位置，不满足的放在_2位置

scala > X = 'start //定义符号（symbol）类型，使用'定义
scala > println(X)  //输出'start
scala > if(s1=='start) println("this") else println("other") //输出“this”，这种在模式匹配、内容判断中比较常见

scala > x = if( "hello" == "hell" ) 1 else 0
x : Int = 0  //if 可以用作表达式使用，返回值可以给变量赋值

for (i<- 1 to 3 ) println("i="+i) //输出i=1，i=2，i=3。<-叫生成器。for实现遍历，scala中不建议用while。1 to 3 可以用 1.to(3)替换，利用方法进行int转型成RichInt。
scala > 1 until (3，1)   //输出1,2，until实现左闭右开,利用括号实现步长（1）控制

for(i <- 1.to(40) if (i%4==0) ; if (i%5==0) )  //没有continue，也很少用break，可以加过滤条件if ，多条件间加；
{
	println("i=" +i )
}

scala > var x = for (i <- 1 to 5 )  yield i/2  //for和yield一起使用可以当成表达式，每循环一次yield就会生成对应的值并保存进缓存，执行完利用缓存生成集合并返回
x : scala.collection.immutable.IndexedSeq[Int] = Vector(0,1,1,2,2)  //1除以2等于0,2除以2等于1...


//数组
scala > var X = new Array[Int] (3)  //使用new定义定长数组array,内容为int型，长度为3，内容为null,null,null。区别于X = Array(3)是生成长度为I，内容是3的数组
scala > X(0) = 1 //数组赋值
scala > var Y = ArrayBuffer(1,2,3) //变长数组为arraybuffer
scala > Y.max //求最大值，同理.sum求和.min求最小
scala > Y.sortWith(_<_) //从小到大排序
scala > Y += 4 //用+=在尾部追加元素。输出ArrayBuffer(1,2,3,4)
scala > Y ++= List (5,6) //用++=可以追加任何集合。输出ArrayBuffer(1,2,3,4,5,6)
scala > Y.trimEnd(5) //删除末尾3个元素。输出ArrayBuffer(1)
scala > Y.insert(1,2,3,4)  //从索引位置1开始，插入2,3,4。输出ArrayBuffer(1,2,3,4)
scala > Y.remove(1,2) //从1这个索引位置（2所在的地方）开始删掉2个元素。输出ArrayBuffer(1,2)
scala > Y.toString() //int型，不能直接用这个转为string，会输出它的位置，例如【I@141aba8
scala > Y.mkString(",")  //使用分隔符，将int内容分割输出，结果res0：string=1,2
scala > Y.mkString( "<" , "," , ">")  //结果res1：string=<1,2>
scala > Y.sortWith(_>_) //由大到小排序
var XX = Array(Array(1,2),Array(3,4))  //创建二维数据
XX(0)(1) //获取0行1列元素，输出2
for(i <- XX)  //使用双循环遍历多维数组，先遍历xx的行，依次赋值给i,所以第一次i=Array(1,2),第二次是Array(3,4)
	for(j <- i) print(j)  //再在i中依次遍历赋值给j，所以当i=Array(1,2)时，j依次取1,2，同理后面。输出1234。


//列表，集合
//对于列表list，本质上推荐使用::添加单个元素，:::添加多个元素，+:等加减号用法纯粹是为了适配其他语言的旧习惯
//对于集合set等，+添加元素，-删除元素，++，--批量添加删除
var list1 = List (1,2,3)  //定义初始化list。或者赋值用List.apply(1,2,3)。
var list2 = 1::2::3::Nil //生成list(1,2,3)，Nil（必须加这个！）表示空list，::表示在list头部追加元素。从右向左看，相当于在空listNil头部加3，再在3前加2...
var list3 = 1+:2+:3+:Nil //生成list(1,2,3)，+:和::相似，只是前者不能用于模糊匹配
var list4 = Nil:+1:+2:+3 //生成list(1,2,3)，:+在尾部加。总之冒号跟在集合那一侧。
var list5 = list1++list2 //生成list(1,2,3,1,2,3)，++用于连接两个集合
var list6 = list1:::list2 //生成list(1,2,3,1,2,3)，:::只能用于连接两个List类型的集合
var list6 = list1::list2  //生成List[java.io.Serializable] = List(List(1, 2，3), 1, 2,3) ，会直接把list1整体放在list2第一个元素的位置。
//Iterable的方法
list1.isEmpty //是否为空，返回布尔值
list1.count(pred)  //返回满足前提表达式的元素计数
list1.forall(pred)  //所有元素都满足pred返回true
list1.exists(pred)  //至少有一个元素满足pred返回true
list1.exists(pred)  //至少有一个元素满足pred返回true
list1.filter(pred)  //返回满足前提表达式的元素
list1.filterNot(pred)  //返回不满足前提表达式的元素
list1.takeWhile(pred)  //返回满足前提表达式的一组元素（直到遇到第一个不满足的元素）
list1.slice(from,to)  //返回位于(from,to) 区间的所有元素
list1.head //1,去第一个元素
list1.tail //list(2,3),除第一个元素以外的其他元素
list1.tail.head //2，第二个元素
list1.last //3，最后一个元素
list1.init //list(1,2)，除最后一个元素以外其他
list1 drop 1 // 返回List(2, 3)。丢弃前n个元素，等效于list1.drop (1)（并不是所有方法都能不加.的这样用）。
list1 take 1  //1.取前n个元素
list1.splitAt(2) //(list(1),list(2,3)),按索引位置分割list
list1.toArray //转为数组
var num = List(1,2)
var char = List("1","2")
num zip char //输出List[(Int),(Char)]=List ((1,1),(2,2))
List.unzip(num zip char) //unzip解成一个拥有两个List元素的元祖
List.flatten(List(List(1,2),List(3))) //铺平成一个list，结果List（1,2,3）
List.concat(List(1,2),List(3)) //结果List（1,2,3）。同++和:::用法
List.range(1,6,2) //输出List(1,3,5),range方法是左闭右开
List.make(3,"hi") //输出List("hi","hi","hi"),make方法用于构建n个相同的元素
//seq序列方法
list.contains(elem)  //如果该序列包含给定元素，返回true
list.containsSlice(seq)  //如果该序列包含给定序列，返回true
list.startsWith(seq)  //如果该序列以给定序列开始，返回true
list.endsWith(seq)  //如果该序列以给定序列开始，返回true
list.indexOf(elem)  //返回首个元素下标，同理list.indexOfSlice(seq)
list.lastIndexOf(elem)  //返回末尾元素下标，同理list.lastIndexOfSlice(seq)
list.indexWhere(pred)  //返回满足表达式的元素下标
list.prefixLength(pred)  //返回满足表达式的最长元素序列长度
list.intersect(seq)  //返回list与seq的交集。同理list.diff(seq)，返回不同的部分
list.reverse //list反转
val list = List( "a", "g", "F", "B", "c")
list.sorted //List(B，F，a，c，g)。使用元素本身大小排序。默认字符串顺序。
list.sortWith(_ < _) //List(B, F, a, c, g)。使用二元函数less大小排序。
list.sortBy(f) //f是带有映射的函数。例如m = Map(-2 -> 5,2 -> 6)则可以使用m.toList.sortBy(_._2)，根据第二个位置value排序（key，value）整体

val studentInfo = Map("john"->21,"lucy"->20) //创建map，默认是immutable类型，不可变，clear()都不可用
val xMap = new scala.collection.mutable.Map[String,Int]() //创建一个mutable的map，或者 以元祖形式 .Map(("spark",1),("hive",1))
xMap.put("spark",1) //使用put函数添加设置元素
xMap+=("spark"->1) //同上，添加元素，同理-=删除元素
xMap.keySet //获取map集合中的所有键
xMap.values //获取map集合中的所有值
xMap.contains("spark") //返回true
xMap.get("spark")  //返回Option[Int] = Some(1),Option(选项)类型用来表示一个值是可选的（有值或无值)。 Option[T]就是一个Some[T] ，反之就是None 
xMap.getOrElse("spark",0)  //获取键所对应的值，若键不存在则返回0.
xMap("spark") = 2 //添加或更新map中的值



def higherOrderFunction(factor:Int) = (x:Double) => factor*x //定义高阶函数（参数和返回值可以是函数）。factor为定义时必须的参数。x为调用时参数。
val multiply = higherOrderFunction(100) //定义高阶函数为变量，factor=100
multiply(10) //输出1000，因为x=10，执行100*10的操作
higherOrderFunction(100)(10) //输出也是1000，同样效果
def multiply2(factor:Int)(x:Double) = factor*x //定义柯里化函数
multiply2(100)(10) //输出1000。但柯里化函数不是高阶函数，不能像higherOrderFunction(100)一样调用multiply2(100)返回函数对象。
val paf = multiply2(100)_ //可以使用_暂时省略参数
paf(10)  //输出1000，调用paf再把第二个参数传入

Array("spark","hive").map(_*2) //结果Array(sparkspark,hivehive），map将函数应用到集合的每个元素。即每个元素都*2。“_”是占位符，因只有一个变量，故可以用
val list = List("spark" -> 1,"hive" -> 2)
list.map(_._1) //结果List(spark,hive)。第一个_是占位符替代元素，_1是访问元祖的第一个元素。

List(1,2,3,4).reduce(_+_) //结果10。每次相加两个元组然后产生新的与下一位相加。即先1+2=3，再3+3=6,再6+4=10。
Array(1,2,3,4).fold(3)(_+_) //结果13,3+1=4；4+2=6；6+3=9；9+4=13.fold(n)中的n是累加的起始数。reduce中默认起始数是0。


val isEven:PartialFunction[Int,String]={case x if x % 2 ==0 => x +" is even" } //定义偏函数，x指代当前元素，满足除以2得以0则执行。
isEven(10)  //输出 10 is even。不满足条件的不执行。
(1 to 10).collect(isEven) //输出Vector(2 is even, 4 is even, 6 is even, 8 is even, 10 is even)

List(1, 3, 5, "seven") map { case i: Int => i + 1 } //case 判断当前元素i是否Int型，是则执行i+1。但这是使用map会报错，因为map在执行前不进行类型判断。
List("a","b").map()
List(1, 3, 5, "seven") collect { case i: Int => i + 1 } //输出List(2, 4, 6)，collect接受偏函数类型的参数，会先判断参数是否在case中定义，再执行

def sum(x:Int*){  //输入参数是一个整形变长序列，使用*表示。没有返回值的函数不需要在{}前加=
	for (i <- x) print(i+" ")  
}
val s = sum(1,2,3) //输出1 2 3

//模式匹配
object TestPat extends App{ //继承APP类后可以不写main函数而直接执行，减少代码复杂度

 	def  patterMatching1 (x:int) = x match { //定义一个函数判断是否输入参数x能被2整除
 		case 1 => "输入的是1" //输入匹配到是1的情况,返回“输入的是1”字符串
 		case i if (i%2==0) =>"能被2整除"  //因为匹配的是一个满足某种条件的未知数，所以需要用变量i表示，并在后面加if条件
 		case x:int => "输入是整形" //进行类型判断
 		case _ => "其他"  //默认其他项
 	}
 	println(patterMatching1(1)) //打印字符串"输入的是1"，虽然也满足是整形的条件，但是会按顺序执行
 	println(patterMatching1(4)) //打印字符串"能被2整除"


 	case class Dog(val name:String,val age:Int) //定义成case类是为了在后面匹配用，因为普通类匹配时需要多加步骤。
 	val dog = Dog("D",2) //创建一个对象
 	def patterMatching2 (x:AnyRef) = x match { //AnyRef表示可以接受任意类型
 		case Dog(_,age) =>println(s"Dog age = $age")  //匹配到输入类型是Dog型，可以把name,age都解析出来，如果不用某一变量可用_替代。加s再使用$后加变量用来指代输出。
 		case _ =>  //返回为空
 	}
 	patterMatching2(dog) //输出Dog age = 2


 	val arr = Array（1,2,3）
 	def patterMatching3(x:AnyRef) = x match {
 		case Array(f,s) => "二维" //匹配到两个维度的Array时
 		case Array(f,_,t) => "三维"  //三个维度，中间不需要的变量可以用_省略
 		case Array(f,_*) => println(s"第一个数是$f") //需要几个变量写几个，后面不要的可以用_*整体省略。但_*只能用在尾部，且只能用于Array,list
 		case _ => 
 	} 
 	patterMatching3(arr) //第一个数是1


 	val tuple x =(1,2,List(3,4,5))
 	def patterMatching4(x:AnyRef) = x match {
 		case (2,_,_) => println("第一个数是2") //元祖匹配时不需要写类型，即直接case（），不用case tuple ().只要第一位是2，就能匹配到该条。不能用_*.
 		case (_,_,e1@List(_*)) => println("第三位是List型"+e1) //@可以将后面的对象赋给前面的参数(e1)
 		case _ => 
 	} 
 	patterMatching4(x)


 	val rgexNum = """[1-9][0-9]*""".r  //定义一个正则表达式，"""包裹，.r转换。* 号代表前面一个字符（[0-9]）可以不出现，也可以出现一次或者多次（0次、或1次、或多次）
 	val rgexEng = """[a-z]+""".r //[a-z]指定字母范围，+号表示前面一个字符必须至少出现一次（1次或多次）。
 	val rgexNor = """colou?r""".r //可以匹配 color 或者 colour，? 问号代表前面的字符最多只可以出现一次（0次、或1次）
 	val rgex1 = """(\d\d\d\d)-(\d\d)-(\d\d)""".r //定义一个日期正则式
 	val rgex2 = new scala.util.matching.Regex("""(\d\d\d\d)-(\d\d)-(\d\d)""","year","month","day") //也可以new一个新的正则式，后面可以跟正则式的group名称
 	val text = "2019-03-04  2019-03-05" //定义待匹配的字符串。（好像需要提前定义，如果在后面rgx.find类函数中直接使用“xx”会没有返回值？）
 	val text2 = "from 2019-03-04 to 2019-03-05" //定义待匹配的字符串
 	for(rgex1(year,month,day) <- rgex1.findAllIn(text)){ //findAllIn用于返回所有匹配字符串。for中可以直接使用模式匹配，简化代码
 		println(s"year=$year,month=$month,day=$day")  //输出year=2019,month=03,day=04  year=2019,month=03,day=05
 	}
 	val rep = rgex2 replaceAllIn (text2, m => m.group("month")+"/"+m.group("day")) //返回from 03/04 to 03/05。即先匹配到"2019-03-04"和"2019-03-05"分别作为m,
 																				   //再将m的group中名为"month"和"day"的部分取出用"/"连接，最后塞回原位置。


 	for((language,framework)<-Map("java"->"hadoop","scala"->"spark")) {  //直接在for中使用模式匹配将map中的元素遍历解析成(language,framework)形式
 		print(s"$framework is developed by $language.")  //输出hadoop is developed by java. spark is developed by scala.
 	}
 	for((language,"spark")<-Map("java"->"hadoop","scala"->"spark")){ //language是变量，"spark"表示具体字符串。即需要匹配第二个位置为"spark"的那一个元素
 		print(s"spark is developed by $language") //输出park is developed by scala
 	}
 	for((language,framework:String)<-Map("java"->"hadoop".length,"scala"->"spark")){ //将framework变量设定为String类型，即只匹配第二个位置为String类型的元素
 		print(s"$framework is developed by $language") //输出park is developed by scala。因为"hadoop".length是int型。
 	}
}

//泛型
object TestVar extends App{
	val astr:Array[String] = Array("spark","hadoop")
	val aint:Array[Int] = Array(1,2)
	def printAll(x:Array[_]){ //x是数组，但数组类型不确定。可以使用泛型[]，用"_"省略泛型表示符。完整可写作x:Array[T] forsome{type:T}
		for(i<=x){
			print(i+" ")
		}
	}
	printAll(astr) //输出spark hadoop 
	printAll(aint) //1 2 

	case class student[S,T<:AnyVal](var name:S,var age:T) //S和T都是泛型，但是T有上界限定，使用<:表示(同理下界>:)。AnyVal包括Int,double等等，但不包括String。
	val S1=student("Nic",12) //调用
}


//读取文件
import scala.io.Source
val source = Source.fromFile("test.txt")  //读取文件
val source2 = Source.fromURL("http://horstamnn.com","UTF-8")  //读取网页
val source3 = Source.fromString("hello world")  //读取文件
val age = Source.stdin  //读取控制台
val lineIterator = source.getLines //获得文件中的每一行，返回的是个迭代器
for(i <- lineIterator) print(i)  //遍历迭代器并输出
val lines = lineIterator.toArray  //将迭代器内容转为数组
val contents = source.mkString  //将文件内容读成字符串
val tokens = source.mkString.Split("\\s+")  //将文件中用空格（\\s+或者直接用空格代替）隔开的内容分割数组
val num = tokens.map(_.toDouble)  //将分割内容转为double型数组
source.close() //使用完需要关闭
//scala没有读取二进制文件的方法，要使用java类库
val file = new File (filename)
val in = new FileInputStream(file)
val bytes = new Array[Byte](file.lengh.toInt) //将文件读成字节数组
in.read(bytes)
in.close()
//写入文本,也是使用java
val out = new PrintWriter("num.txt")
for(i <- 1 to 100) out.println(i)
out.print("%6d %10.2f".format(a,b)) //使用print方法时可能会遇到需要转为AnyRef的问题，可以使用a.asInstanceOf[AnyRef]或使用string类的format方法
out.close()
//访问目录
import java.io.File
//遍历某个目录下所有子目录的函数
def subdirs(dir:File):Iterator[File] = {  //输入dir是个文件类型。返回是文件类型的迭代器
	val children = dir.listFiles.firlter(_.isDirectory)
	children.toIterartor ++ children.toIterartor.flatMap(subdirs _)  //
}
for(d<-subdirs(dir)) //使用该语句调用


//scala脚本 
//进程库使用人们熟悉的shell操作符I > >> < && ||，只不过给它们加上了#前缀
import sys.process._
"ls -al .." ! //Is -al ..命令被执行，显示上层目录的所有文件。执行结果被打印到标准输出。!操作符返回的结果是被执行程序的返回值：程序成功为0，为非0。
val result = "ls -al .."!! //使用! !操作符而不是!操作符的话，输出会以字符串的形式返回
"ls -al .." #> new File("output.txt") ! //要把输出重定向到文件，使用#>操作符
"ls -al .." #>> new File ("output.txt") ! //要追加到文件末尾而不是从头覆盖的话，使用#>>操作符
"grep sec" #< new File("output.txt") ! //要把某个文件的内容作为输入，使用#<操作符


