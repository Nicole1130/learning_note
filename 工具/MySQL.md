## 一、安装：  
    1.https://dev.mysql.com/downloads/mysql/ 下载合适电脑配置的zip包。点击download后不需要注册，直接选左下角“No thanks, just start my download”；  
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
    4.以管理员身份打开cmd命令窗口。输入：mysqld --initialize --user=mysql --console。如报错可能是缺少依赖文件。成功则会显示初始密码，需记录下来后续使用。  
        *注 vcruntime140_1.dll链接: https://pan.baidu.com/s/1-46kOeYmjF6i4at1sHL0uA 提取码: m52r。  
    5.输入：mysqld install。显示成功后可验证【任务管理器】中【服务】中是否有mysql服务，此时因处于关闭状态。  
    6.输入：net start mysql。成功应显示服务已开启。
    7.服务开启后，命令行输入： mysql -u root -p 跳出输入密码提示，输入先前记录的初始密码。出现Welcome则表示已成功安装mysql。  
    8.在命令行mysql-> 后输入：set password='root'; （注意分号需输入）设置账户密码为root。  
    9.命令行输入：quit 。退出。  
    10.下载破解版数据库可视化工具Navicat，https://www.52pojie.cn/thread-952490-1-1.html。按操作说明破解安装。  
    11.开启Navicat，【文件】-【新建连接】中输入连接名称、主机ip、用户名和密码。【测试连接】并点击【确定】完成。  
## 二、idea中创建springboot链接MySQL
    1.新建项目时除了勾选相应依赖外，还需添加【SQL】中【mysql driver】 和【mybatis framework】。
    2.打开项目后，右侧找到【database】选择【mysql】。输入name（样式为：库名@localhost）、用户名user和密码password、库名database、url（样式为jdbc:mysql://localhost:3306/库名）。如下方提示没有driver则点击下载。
    3.点击test connection链接，如成功应显示绿勾和mysql版本信息。
        > *若失败，提示Server returns invalid timezone. Go to 'Advanced' tab and set 'serverTimezone' property manually.说明mysql时区未设    置。解决方案见https://blog.csdn.net/ITMan2017/article/details/100601438 或者（1）进入命令窗口（Win + R），连接数据库 mysql -hlocalhost -uroot -p，回车，输入密码，回车，进入mysql.（2）输入 show variables like'%time_zone'; （注意分号需输入），回车，第二项中显示 SYSTEM 表明确实没有设置时区。（3）输入set global time_zone = '+8:00'; 设置成功即可返回idea重新测试连接。
    
