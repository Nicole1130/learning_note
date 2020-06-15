## 一、安装：  
1.https://dev.mysql.com/downloads/mysql/ 下载合适电脑配置的zip包。点击download后不需要注册，直接选左下角“No thanks, just start my download”;  
2.解压，添加.bin目录到环境变量；  
3.mysql根目录文件下面新建一个mysql.ini和一个data文件夹。mysql.ini内容如下：

        [mysql]  
        # 设置mysql客户端默认字符集  
        default-character-set=utf8   
        [mysqld]  
        #设置3306端口  
        port = 3306   
        # ！！！！！！！！！！！！设置mysql的安装目录！！！！！！！！！！！  
        basedir=D:\Nicole\software\mysql-8.0.20-winx64  
        # ！！！！！！！！！设置mysql数据库的数据的存放目录！！！！！！！！  
        datadir=D:\Nicole\software\mysql-8.0.20-winx64\data  
        # 允许最大连接数  
        max_connections=200  
        # 服务端使用的字符集默认为8比特编码的latin1字符集  
        character-set-server=utf8  
        # 创建新表时将使用的默认存储引擎  
        default-storage-engine=INNODB  
		
4.以管理员身份打开cmd命令窗口。输入：**mysqld --initialize --user=mysql --console** 如报错可能是缺少依赖文件。成功则会显示初始密码，需记录下来后续使用。  
> 如缺少 vcruntime140_1.dll 下载链接为: https://pan.baidu.com/s/1-46kOeYmjF6i4at1sHL0uA 提取码: m52r。  

5.输入：**mysqld install** 显示成功后可验证【任务管理器】中【服务】中是否有mysql服务，此时因处于关闭状态。  
6.输入：**net start mysql** 成功应显示服务已开启。  
7.服务开启后，命令行输入： **mysql -u root -p** 跳出输入密码提示，输入先前记录的初始密码。出现Welcome则表示已成功安装mysql。  
8.在命令行mysql-> 后输入：**set password='root';** （注意分号需输入）设置账户密码为root。  
9.命令行输入：**quit**  退出。  
10.下载破解版数据库可视化工具Navicat，https://www.52pojie.cn/thread-952490-1-1.html 按操作说明破解安装。  
11.开启Navicat，【文件】-【新建连接】中输入连接名称、主机ip、用户名和密码。【测试连接】并点击【确定】完成。  
## 二、idea中创建springboot链接MySQL  
> 需了解的背景知识：  
> * Spring框架是Java平台上的一种开源应用框架，SpringBoot是2014年开源的轻量级框架。目的是通过配置来进一步简化Spring应用的整个搭建和开发过程。  
> * JDBC全称Java Database Connectivity，是Java语言中用来规范客户端程序如何来访问数据库的应用程序接口。  
> * MyBatis是对JDBC的进一步封装，是一个支持普通SQL查询，存储过程和高级映射的优秀持久层框架。它消除了几乎所有的JDBC代码和参数的手工设置以及对结果集的检索封装。可以使用简单的XML或注解，将sql语言写至项目的Java文件中，完成与数据库的交互。  

1.新建项目时除了勾选相应依赖外，还需添加【SQL】中【mysql driver】【JDBC】 和【mybatis framework】。  
2.打开项目后，右侧找到【database】选择【mysql】。输入name（样式为：库名@localhost）、用户名user和密码password、库名database、url（样式为jdbc:mysql://localhost:3306/库名） 。如下方提示没有driver则点击下载。  
3.点击test connection链接，如成功应显示绿勾和mysql版本信息。  
> 若失败，提示Server returns invalid timezone. Go to 'Advanced' tab and set 'serverTimezone' property manually.说明mysql时区未设    置。解决方案见https://blog.csdn.net/ITMan2017/article/details/100601438 或者（1）进入命令窗口（Win + R），连接数据库 mysql -hlocalhost -uroot -p，回车，输入密码，回车，进入mysql.（2）输入 show variables like'%time_zone'; （注意分号需输入），回车，第二项中显示 SYSTEM 表明确实没有设置时区。（3）输入**set global time_zone = '+8:00';** 设置成功即可返回idea重新测试连接。  

4.打开application.properties添加  

```
	# 3306是mysql默认端口，test是库名称
	spring.datasource.url=jdbc:mysql://127.0.0.1:3306/test 
	# 设置用户名和密码
	spring.datasource.username=root
	spring.datasource.password=root
	spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
	spring.datasource.max-idle=10
	spring.datasource.max-wait=10000
	spring.datasource.min-idle=5
	spring.datasource.initial-size=5
	jdbc.DriverClassName=com.mysql.cj.jdbc.Driver
```
如使用maven可能还需的依赖，在pom.xml中添加并等待它自动下载更新  
```
        <!--mysql和JDBC-->
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-jdbc</artifactId>
        </dependency>
        <!-- MyBatis-->
        <dependency>
            <groupId>org.mybatis.spring.boot</groupId>
            <artifactId>mybatis-spring-boot-starter</artifactId>
            <version>1.3.2</version>
        </dependency>
		<dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <!-- thymeleaf是Spring官方支持的服务渲染模板（方便HTML中编写变量） -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-thymeleaf</artifactId>
        </dependency>
        <!-- 使用html的依赖 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
            <exclusions>
                <exclusion>
                    <groupId>org.junit.vintage</groupId>
                    <artifactId>junit-vintage-engine</artifactId>
                </exclusion>
            </exclusions>
        </dependency>
```
5.根目录下自动生成的XXXApplication类为springboot的入口启动文件，运行该文件启动进程。  
6.为了在项目程序中连接数据库，新建5个java class(准确说其中2个为接口)，分别为  
* XXXController——class类，用来向外部访问提供接口映射的函数，即当外部请求访问所设定的网页地址时，服务器将采取的措施。  
* XXXService——interface接口，只用来定义服务器端提供功能的函数。  
* XXXServiceImpl——class类，是服务器端提供功能的具体函数实现。  
* XXXMapper——interface接口，用来将Java函数与sql语言做关联的接口。  
* XXXEntity——class类，用来将数据库中元素定义为java类的函数，数据库中每一列对应该java对象的一个属性。  
> 内部具体代码demo可参考https://www.jianshu.com/p/ca185e2b19fe  
7.需要注意的是，为了解决@Autowired的注入问题，应在入口文件XXXApplication类中把新建的文件路径全部引入，可以使用他们共同的父目录。缺少路径会在启动时报错，提示找不到相应的bean。在import后加入代码：  
```
import org.mybatis.spring.annotation.MapperScan;
@MapperScan("路径")
```
8.重新启动项目！重新启动项目！重新启动项目！  
9.运行XXXApplication，在浏览器中输入**localhost:8080/你在Controller中设置的跳转地址**，完成。
