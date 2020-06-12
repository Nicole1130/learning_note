一、安装：
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
    4.以管理员身份打开cmd命令窗口。输入：mysqld --initialize --user=mysql --console。如报错可能是缺少依赖文件。成功则会显示初始密码，需记录下来后续使用。。
       *vcruntime140_1.dll链接: https://pan.baidu.com/s/1-46kOeYmjF6i4at1sHL0uA 提取码: m52r。
    5.以管理员身份打开cmd命令窗口。输入：mysqld install。显示成功后可验证【任务管理器】中【服务】中是否有mysql服务，此时因处于关闭状态。
    6.以管理员身份打开cmd命令窗口。输入：net start mysql。成功应显示服务已开启。
    7.服务开启后，命令行输入： mysql -u root -p 跳出输入密码提示，输入先前记录的初始密码。出现Welcome则表示已成功安装mysql。
    8.在命令行mysql-> 后输入：set password='root'； 。设置账户密码为root。
    9.命令行输入：quit 。退出。
    10.下载破解版数据库可视化工具Navicat，https://www.52pojie.cn/thread-952490-1-1.html。按操作说明破解安装。
    11.开启Navicat，【文件】-【新建连接】中输入连接名称、主机ip、用户名和密码。【测试连接】并点击【确定】完成。
    
