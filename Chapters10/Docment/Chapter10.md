# 10.1 SQLFlow简介


## 10.1.1 什么是SQLFlow

科技总是随着时代在无时无刻的进步着. 作为近10年内极具代表性的技术, 人工智能在社会发展的各行各业中都得到了极为广泛的应用. 无论是图像处理技术在安全监控, 自动驾驶领域内的成功应用, 还是自然语言处理技术在智能客服, 内容生成等领域上获得的巨大进步, 亦或是深度强化学习在围棋领域击败一众顶尖强者, 都预示着人工智能技术将在可见的未来给人类社会的发展进步带来无限的可能.

与此同时, 在人工智能技术推进工业发展的过程中也面临着一系列的实际问题. 其中一个常见而又无法忽视的问题就是深入了解和掌握机器学习技术需要极为丰富的知识储备, 例如高等数学、统计学、概率论、以及熟练的编程技能. 与此同时, 为了保证能够技术在业务中的切实落地, 在两者的融合过程中还需要专家具备对业务逻辑的深度理解, 方能够保证技术可以带来真实可靠的业务提升效果而非空中楼阁. 

这些要求无疑提高了机器学习结合业务赋能业务的门槛, 同时也制约着整个人工智能产业的长远发展. 而现在具备容易上手, 方便使用特点的SQLFlow正是为解决上述问题而来. 

SQLFlow是由滴滴数据科学团队和蚂蚁金服合作开源的一款链接数据和机器学习能力的分析工具, 旨在能够抽象出从数据到模型的研发过程, 同时配合底层的引擎适配及自动优化技术, 使得具备基础SQL知识的技术人员也可以完成大部分的机器学习模型训练,预测及应用任务. 

SQLFlow希望通过简化和优化整个研发过程来将机器学习的能力赋予业务专家, 从而推动更多的人工智能应用场景被探索和使用.

## 10.1.2 SQLFlow的定位和目标

将SQL与AI连接起来的这个想法并非SQLFlow独创. Google在2018年年中发布的BigQueryML, TeraData的SQL for DL, 以及微软基于SQL Server的AI扩展, 同样旨在打通数据与人工智能之间的连接障碍, 使数据科学家和分析师们能够通过SQL语言就可以实现机器学习功能并完成数据预测和分析任务. 

但是与上述各个系统不同的是, SQLFlow着力于链接更广泛的数据引擎和人工智能技术框架, 而并不仅仅局限于某个公司产品内的封闭技术栈. 更为重要的是, 这是一个面向全世界的开发者的开源项目, 任何对这一领域感兴趣的开发者都可以参与其中, 项目的组织者们也希望能够借助所有开发者和使用者的力量来一起共同建设社区, 促进这一领域的健康发展.

作为连接数据引擎和AI引擎的鹊桥, SQLFlow目前支持的数据引擎包括MySQL, Hive和MaxCompute, 支持的AI引擎则不仅包括业界最流行的TensorFlow, 还包括XGBoost, Scikit-Learn等传统机器学习框架.

| 数据引擎 | 简介 | 链接 |
|:---:| :---| :---|
| MySQL | MySQL是由瑞典MySQL AB公司开发的关系型数据库管理系统, 目前属于Oracle旗下产品, 是目前应用最广泛的关系型数据库管理系统(RDBMS, Relational Database Management System)之一. | https://www.mysql.com/ |
| Hive | Hive是基于Apache Hadoop构建的数据仓库分析系统, 通过Hive SQL提供了丰富的查询方式来分析存储在分布式文件系统中的数据. | https://hive.apache.org/ |
| MaxCompute | MaxCompute是一种快速、完全托管的TB/PB级数据仓库解决方案, 面向用户提供了完善的数据导入方案以及多种经典的分布式计算模型, 能够帮助用户更快速的解决海量数据计算问题, 有效降低企业成本并保障数据安全. | https://www.aliyun.com/product/odps |


| AI引擎  | 简介 | 链接 |
| :---: | :---| :--- |
| TensorFlow | TensorFlow是谷歌出品的端到端开源机器学习平台, 拥有一个包含强大工具和丰富社区资源的富有生命力的生态系统, 同时兼顾研究领域和工业环境的需求, 是当下最强大的深度学习框架之一.  | https://www.tensorflow.org/ |
| Scikit-Learn | Scikit-learn 是基于Python的开源机器学习库, 基于NumPy和SciPy等科学计算库实现, 并支持支持向量机、随即森林、梯度提升树、K均值聚类等多种机器学习算法. | https://scikit-learn.org/ |
| XGBoost | XGBoost(Extreme Gradient Boosting)是Gradient Boosting Machine的C++实现, 能够自动利用CPU的多线程进行并行, 同时在算法上加以改进以提升精度效果. | https://xgboost.ai/ |
| SHAP | Shaply Value是由美国洛杉矶加州大学教授罗伊德·夏普利(Lloyd Shapley)提出, 用于解决合作博弈的贡献和收益分配问题. SHAP(SHapley Additive exPlanation)是在合作博弈论的启发下, 通过计算模型预测值在各特征之间分配的SHAP VALUE评估特征重要性的模型解释工具. | https://github.com/slundberg/shap |

## 10.1.3 SQLFlow工作原理

我们以Docker镜像中的`Iris`案例来说明SQLFlow的工作原理. 

Iris数据集, 也称鸢尾花数据集, 是机器学习领域最为常用的分类实验数据集之一, 首次出现在著名的英国统计学家和生物学家Ronald Fisher 1936年发表的论文《The use of multiple measurements in taxonomic problems》中, 被用来介绍线性判别式分析. 

该数据集内包含150个数据样本, 分为3类, 每类中包含50个样例. 每条样例记录包含花萼长度, 花萼宽度, 花瓣长度和花瓣宽度4个属性. 数据集的目标是通过这4个属性来预测鸢尾花卉样例记录属于Setosa，Versicolour，Virginica三个种类中的哪一类.

为了简便起见, 我们预先将数据存储至`iris.train`表中. 前四列表示训练样例的特征, 最后一列代表训练样本的标签.

分类模型以`DNNClassifier`为例. `DNNClassifier`默认具有双隐藏层, 每层的隐藏节点数均为10, 分类数为3, 默认优化器和学习率分别为`Adagrad`和0.1, 损失函数则默认配置为`tf.keras.losses.sparse_categorical_crossentropy`.

从`iris.train`表中获取数据并训练对应`DNNClassifier`模型的训练语句如下:

```sql
SELECT * FROM iris.train
TO TRAIN DNNClassifer
WITH hidden_units = [10, 10], n_classes = 3, EPOCHS = 10
COLUMN sepal_length, sepal_width, petal_length, petal_width
LABEL class
INTO sqlflow_models.my_dnn_model;
```

SQLFlow会将上述接收到的SQL段落进行解析, 其中`SELECT`语句传递给对应数据引擎进行执行来获取数据, 而`TRAIN`和`WITH`语句则分别指定了使用的模型种类, 模型结构和训练所需的超参数. `COLUMN`和`LABEL`部分则分别用于指定使用的特征所需要的数据各列名称.

经过上述解析, SQLFlow将上述SQL段落解析为对应的`Python`程序.

<div align="center">
    <img src="./pics/10.1_SQLFlow_workflow.png" width="800" alt="Describe Clause">
    <p>图10.1 SQLFlow工作原理</p>
</div>


# 10.2 SQLFLow 运行环境的设置

SQLFlow为了方便用户使用, 在Docker Hub中维护有基于最新代码的相关镜像. 镜像中默认集成MySQL作为数据存储引擎, Jupyter Notebook作为前端使用界面. 通过简单的命令就可以启动一个SQLFlow个人环境, 用户可以在其中进行尝试和熟悉使用命令, 同时也可以通过设置不同配置以适应工业环境需求.

## SQLFlow in Docker

Docker是DotCloud公司于2013年3月以Apache 2.0授权协议开源的容器引擎, 旨在帮助开发者实现更轻量级的环境迁移, 更便捷的持续交付和部署, 以及更轻松的维护和扩展.

开发者可以在容器中打包应用及所需依赖并且发布和部署到任何机器上. 得力于Docker使用的分层存储和镜像技术, 用户可以在镜像的不同层之间复用重复部分, 从而使得镜像的管理和维护变得更加简单.

1. 安装Docker

    安装Docker的细节超出了本书的范畴, 读者可以参考[Docker Community Edition](https://docs.docker.com/install/)根据自己的实际情况进行安装.

2. 拉取镜像

    Docker安装就绪后, 需要通过命令`docker pull sqlflow/sqlflow`拉取SQLFlow最新镜像, 默认标签为`latest`.

    <div align="center">
        <img src="pics/10.2_docker拉取SQLFlow最新镜像.png" width="800" alt="Describe Clause">
        <p>图10.2 拉取Docker Hub上最新SQLFlow镜像, 默认标签为latest.</p>
    </div>

3. 创建容器

    拉取镜像完成后, 用户可以通过执行命令 `docker run -it -p 8888:8888 sqlflow/sqlflow` 启动容器, 默认端口为8888. 

    <div align="center">
        <img src="pics/10.3_docker启动容器.png" width="800" alt="Describe Clause">
        <p>图10.3 依据镜像启动容器, 默认端口为8888.</p>
    </div>
    
    用户在等待容器启动完成后, 可以在浏览器中打开`http://localhost:8888/tree?`进入Jupyter主页面. 在主页面中可以看到, 镜像中已经内嵌了部分案例供用户尝试使用.
    
    <div align="center">
        <img src="pics/10.4_SQLFlow_jupyter_homepage.png" width="800" alt="Describe Clause">
        <p>图10.4 容器启动后, 通过浏览器打开Jupyter主页.</p>
    </div>

4. 案例讲解

    这里我们以`iris-dnn.ipynb`为案例进行简单的使用说明介绍. 该案例旨在通过训练`DNNClassifier`模型来实现在`iris`数据集上的分类效果.
    
    步骤一 : 表结构探查
    
    <div align="center">
        <img src="pics/10.5_describe.png" width="800" alt="Describe Clause">
        <p>图10.5 执行DESCRIBE语句, 获取数据库中表相关元信息.</p>
    </div>
    
    步骤二 : 训练数据探查
    
    `SELECT...FROM...LIMIT...`为标准SQL语句, SQLFlow Server在判定该语句为标准语句后会透传给后端的数据引擎进行执行.
    
    通过该语句可以大致了解表中数据格式, 取值分布等情况, 便于及早发现可能存在的数据格式种类问题(例如字符串表示的数字)或异常值, 空值等问题.
    
    <div align="center">
        <img src="pics/10.6_sample_train.png" width="800" alt="Describe Clause"/>
        <p>图10.6 通过SELECT语句进行数据探查.</p>
    </div>
    
    步骤三 : 训练模型
    
    `TO TRAIN`语句用于描述需要训练的模型种类. 用户可以在这里声明需要使用的库和模型种类.
    
    `WITH`语句用于指定模型训练过程中需要的模型参数和相关超参数, 例如模型隐层数(layers), 隐层节点数(hidden units), 优化器(optimizer), 损失函数(loss function), 批大小(batch_size), 迭代次数(epochs)等. 具体的参数需要配合模型的具体种类进行声明和使用.
    
    `COLUMN`语句用于指定参与训练的表中数据列各列名称.
    
    `LABEL`语句用于指定在监督学习中所需的标签列名称.
    
    `INTO`语句用于指定训练完成后将模型存储在数据引擎中所需的表名.
    
    <div align="center">
        <img src="pics/10.7_train.png" width="800" alt="Describe Clause"/>
        <p>图10.7 根据训练数据集进行模型预测.</p>
    </div>
    
    步骤四 : 模型预测
    
    `TO PREDICT`语句用于指定预测结果所需写入的表名和预测的列名.
    
    `USING`语句用于指定进行预测所需的模型.
    
    <div align="center">
        <img src="pics/10.8_predict.png" width="800" alt="Describe Clause"/>
        <p>图10.8 使用模型进行预测.</p>
    </div>
    
    步骤五 : 预测结果探查
    
    通过`SELECT`语句查看预测结果.
    
    <div align="center">
        <img src="pics/10.9_sample_prediction.png" width="800" alt="Describe Clause"/>
        <p>图10.9 预测结果探查.</p>
    </div>
    
除此之外, `sqlflow/sqlflow:latest`镜像中还集成了其他的样例.

- 基于`IMDB Movie Review`数据集的`imdb-stackedbilstm`样例
- 基于`Boston Housing`数据集的`housing-xgboost`模型训练/预测样例
- 基于`Boston Housing`数据集的`housing-analyze`模型可解释性样例
- 基于`Credit Card Fraud Detection`数据集的`fraud-dnn`模型训练样例.

<div align="center">
    <img src="pics/10.10_other_examples.png" width="800" alt="Describe Clause"/>
    <p>图10.10 镜像中集成的其他样例.</p>
</div>

通过`Docker`使用`SQLFlow`的详细介绍可以参考: [Run SQLFlow Using Docker](https://sql-machine-learning.github.io/sqlflow/doc/run/docker/)


## 环境配置

基于Docker使用SQLFlow虽然方便快捷, 但也会部分用户想系统了解SQLFlow的想法. 这一小节的内容用于介绍如何逐步配置环境, 从使用者的角度帮助读者理解SQLFlow的内部流程. 这里以`Ubuntu 16.04`作为案例环境.

1. 更新系统软件列表, 安装相关工具

    Ubuntu系统中`/etc/apt/sources.list`文件用于保存软件更新的源服务器地址. 由于网络延迟较大, 用户访问量多, 国内用户通过默认源服务器进行软件更新和安装的速度一般较慢, 建议用户可以通过在`sources.list`中添加其他可靠源地址(例如公共镜像源地址或公司内镜像仓库地址)来加快更新安装. 这里以阿里云镜像源地址为例.


    ```bash
    sudo echo '\n\
    deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse    
    ' > /etc/apt/sources.list
    ```
    
    更新`sources.list`文件后需通过`apt-get update`命令使得更新生效, 同时更新软件源中软件列表信息. 更新完成后安装`curl`等相关工具.
    
    ```
    sudo apt-get update
    sudo apt-get install -y curl wget bzip2 unzip git
    ```

2. 安装`protobuf`

    Protocol Buffers是一种轻便高效的结构化数据存储格式, 支持包含Java, Python, Go, C++在内的多种语言, 在解析速度上比XML更快, 同时序列化后占用空间更小, 扩展性和兼容性更加优良.
    
    SQLFlow目前依赖protobuf 3.7.1版本, 用户可以根据自己的系统环境在[https://github.com/protocolbuffers/protobuf/releases/tag/v3.7.1](https://github.com/protocolbuffers/protobuf/releases/tag/v3.7.1)下载对应的安装包.

    基于Ubuntu环境我们选择`protoc-3.7.1-linux-x86_64.zip`进行下载, 下载完成后解压缩至`/usr/local`目录下.

    ```bash
    wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protoc-3.7.1-linux-x86_64.zip
    sudo unzip -qq protoc-3.7.1-linux-x86_64.zip -d /usr/local
    rm protoc-3.7.1-linux-x86_64.zip
    ```

3. 配置`Java`和`Maven`环境

    因为后续步骤中需要进行解析器的编译, 所以用户需要先设置相关环境变量`JAVA_HOME`和`MAVEN_OPTS`并进行相关安装.

    ```bash
    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
    export MAVEN_OPTS="-Dorg.slf4j.simpleLogger.log.org.apache.maven.cli.transfer.Slf4jMavenTransferListener=warn"
    sudo apt-get install -y openjdk-8-jdk maven
    ```

4. 配置`Python`环境

    conda是一款开源的软件包管理系统及软件管理系统, 最初被设计用来帮助数据科学家解决令人头疼的包管理和环境管理问题, 后经过发展逐渐成为当前主流的`Python/R`包管理软件, 目前支持`Windows`, `macOS`和`Linux`等多个系统. conda虽然是使用`Python`编写的, 但它可以管理包括`Python`在内的任何语言编写的项目. 用户可以利用conda进行环境管理, 环境中相关依赖的安装, 运行和更新, 并且可以迅速切换不同的环境来满足不同的开发需求.

    Miniconda是conda的一个小型的发行版, 相比较Anaconda所拥有的全面而略带复杂庞大的功能而言, Miniconda只包含了conda, Python和一些常用的工具, 包含pip, zlib等, 从而减少了用户安装时间, 提高了使用的自由度.

    这里推荐使用Miniconda为SQLFlow构建所需的Python环境.

    ```bash
    curl -sL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o mconda-install.sh
    bash -x mconda-install.sh -b -p miniconda
    rm mconda-install.sh
    ```

    通过`Miniconda`创建名为`sqlflow-dev`的虚拟环境, 依赖Python版本3.6.
    
    ```bash
    /miniconda/bin/conda create -y -q -n sqlflow-dev python=3.6
    echo ". /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
    echo "source activate sqlflow-dev" >> ~/.bashrc
    ```

    启动创建好的`sqlflow-dev`环境并安装相关依赖.
    
    ```bash
    source /miniconda/bin/activate sqlflow-dev
        python -m pip install \
        numpy==1.16.1 \
        tensorflow==2.0.0b1 \
        mysqlclient==1.4.4 \
        impyla==0.16.0 \
        pyodps==0.8.3 \
        jupyter==1.0.0 \
        notebook==6.0.0 \
        sqlflow==0.7.0 \
        pre-commit==1.18.3 \
        dill==0.3.0 \
        shap==0.30.1 \
        xgboost==0.90 \
        pytest==5.3.0
    ```

5. 配置`Go`环境

    设置`GOPATH`环境变量, 获取对应go安装包并解压至`/usr/local`路径下.
    
    ```bash
    export GOPATH=/root/go
    export PATH=/usr/local/go/bin:$GOPATH/bin:$PATH
    curl --slient https://dl.google.com/go/go1.13.4.linux-amd64.tar.gz | tar -C /usr/local -xzf -
    ```
    
    完成Go配置后, 通过`go get`命令获取`golint`等相关依赖.
    
    ```go
    go get github.com/golang/protobuf/protoc-gen-go
    go get golang.org/x/lint/golint
    go get golang.org/x/tools/cmd/goyacc
    go get golang.org/x/tools/cmd/cover
    go get github.com/mattn/goveralls
    cp $GOPATH/bin/* /usr/local/bin/
    ```
6. 安装`Jupyter Notebook`

    设置环境变量`IPYTHON_STARTUP`, 创建对应目录并加载SQLFlow magic command. 创建`/workspace`目录作为Jupyter启动根目录, 用户也可以根据自己的情况选择合适的目录.
    
    ```bash
    export IPYTHON_STARTUP=/root/.ipython/profile_default/startup/
    sudo mkdir -p ${IPYTHON_STARTUP}
    sudo mkdir -p /workspace
    ```
    
    安装`SQLFlow magic command`
    
    ```bash
    sudo echo 'get_ipython().magic(u"%reload_ext sqlflow.magic")' >> ${IPYTHON_STARTUP}/00-first.py
    sudo echo 'get_ipython().magic(u"%reload_ext autoreload")' >> ${IPYTHON_STARTUP}/00-first.py
    sudo echo 'get_ipython().magic(u"%autoreload 2")' >> ${IPYTHON_STARTUP}/00-first.py
    ```

7. 配置`SQLFlow`

    在`$GOPATH/src`目录下创建子目录`sqlflow.org`, 从Github上克隆代码仓库至本地. 

    ```bash
    mkdir -p ${GOPATH}/src/sqlflow.org/
    git clone https://github.com/sql-machine-learning/sqlflow.git
    ```

    `go generate`命令用于根据特殊注释生成部分代码, `go install`命令用于生成包文件和可执行文件.

    ```bash
    cd ${GOPATH}/src/sqlflow.org/sqlflow
    go generate ./...
    go install -v ./...
    ```

    将编译生成的`sqlflowserver`和`repl`移动至`/usr/local/bin`路径下. 将`sqlflow_submitter`拷贝至虚拟环境依赖路径下.
    
    ```bash
    sudo mv ${GOPATH}/bin/sqlflowserver /usr/local/bin
    sudo mv ${GOPATH}/bin/repl /usr/local/bin
    cp -r ${GOPATH}/src/sqlflow.org/sqlflow/python/sqlflow_submitter /miniconda/envs/sqlflow-dev/lib/python3.6/site-packages/
    ```
    
    编译打包解析器`parser`并拷贝至`/opt/sqlflow/parser`目录下.
    
    ```bash
    cd ${GOPATH}/src/sqlflow.org/sqlflow/java/parser
    mvn -B clean compile assembly:single
    mkdir -p /opt/sqlflow/parser
    cp ${GOPATH}/src/sqlflow.org/sqlflow/java/parser/target/parser-1.0-SNAPSHOT-jar-with-dependencies.jar /opt/sqlflow/parser
    ```

8. 安装模型库

    `models`是SQLFlow提供的预制模型库. 执行训练任务时, 用户可以从预制模型库中选择已有模型, 而无需从头编写. 同时开发者也可以贡献自己开发的模型至库中供其他用户使用.

    通过预制模型库可以提高使用者的开发速度, 同时避免了复杂的模型实现和效果验证的重复工作.

    `models`可以通过`pip install sqlflow`进行安装, 也可以将`sql-machine-learning/models`仓库代码拷贝至本地进行安装.
    
    ```bash
    git clone https://github.com/sql-machine-learning/models.git
    cd models 
    python setup.py install
    rm -rf models
    ```

9. 配置数据引擎

    安装MySQL的具体步骤细节超出了本书的内容范畴, 读者可以自行检索相关资料.
    
    SQLFlow支持多种数据引擎, 在SQLFlow官网[sqlflow.org](https://sqlflow.org/)给出了详细的配置步骤, 请参考以下链接.
    
    | 数据引擎  |  相关指导 |
    |---|---|
    | MySQL | [https://sql-machine-learning.github.io/sqlflow/doc/run_with_mysql/](https://sql-machine-learning.github.io/sqlflow/doc/run_with_mysql/) |
    | Hive | [https://sql-machine-learning.github.io/sqlflow/doc/run_with_hive/](https://sql-machine-learning.github.io/sqlflow/doc/run_with_hive/)  |
    | MaxCompute  | [https://sql-machine-learning.github.io/sqlflow/doc/run_with_maxcompute/](https://sql-machine-learning.github.io/sqlflow/doc/run_with_maxcompute/)  |
    
    SQLFlow在Git仓库代码`doc/datasets`目录下提供了10.2章节中提到的案例对应的数据文件, 方便用户快速导入相关数据. 其中`create_model_db.sql`用于创建数据库`sqlflow_models`, 以`popularize`为前缀的其他sql文件分别用于将对应的数据导入至数据库中.

以上各步骤完成后, 可以通过`sqlflowserver`命令启动SQLFlow服务器, 默认端口为50051.

## 交互

完成SQLFlow的相关环境配置后, 用户可以分别使用`Jupyter Notebook`和`REPL`的方式来与SQLFlow服务器进行交互.


### Jupyter Notebook

Jupyter Notebook是数据科学家和分析师们喜爱的试验想法的环境, 在这里不仅用户可以随意编写和执行代码, 还可以进行高效的可视化和文档编辑, 非常适合用户进行数据清理和转换, 统计建模, 文档编写等等一系列工作.

Jupyter Notebook可以通过以下命令启动, `--ip`及`--port`分别用来指定用户访问用户访问Jupyter的IP地址和端口. 启动前需要配置`SQLFLOW_SERVER`环境变量来声明待链接的SQLFlow服务器IP地址和端口.

```bash
SQLFLOW_SERVER=localhost:50051
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''
```

启动成功后, 用户可以在浏览器中输入[http://localhost:8888](http://localhost:8888)来进行访问.

### REPL

数据科学家和分析师门在Jupyter Notebook中完成日常的探索和研究工作后, 需要部署模型到工业环境中来执行真正的预测任务, 而工业环境则更期待简洁的交互方式来方便调用服务. 为了方便与任务调度系统进行配合, SQLFlow同样提供了强大的命令行工具`REPL`.

`REPL`工具能够帮助用户在本地更轻松的进行调试和性能分析, 而不用额外启动SQLFlow服务器和Jupyter Notebook, 使用上更加轻量级. 使用SQLFlow镜像的用户也可以在镜像中使用这一工具.

```bash
docker run -it --run --net=host sqlflow/sqlflow repl --datasource="mysql://root:root@localhost:3306/?maxAllowedPacket=0"
```

执行完成后, 看到命令行中出现以下输出就可以使用`REPL`了.

```
Welcome to SQLFlow. Commands end with ;

sqlflow >
```

训练数据探查:

```sql
sqlflwo> SELECT * FROM iris.train LIMIT 1;
+--------------+-0-----------+--------------+-------------+-------+
| SEPAL LENGTH | SEPAL WIDTH | PETAL LENGTH | PETAL WIDTH | CLASS |
+--------------+-0-----------+--------------+-------------+-------+
|         6.4  |         2.8 |          5.6 |         2.2 |     2 |
+--------------+-0-----------+--------------+-------------+-------+
```

| 参数 | 含义 | 样例 |
|---|---|---|
| -e <quoted-query-string> | 不进入交互模式情况下执行训练语句. | e.g. `-e "SELECT * FROM iris.train TO TRAIN DNNClassifier..."`|
| -f <filename> | 不进入交互模式情况下执行训练文件. | e.g. `-f ./my_sqlflow.sql` |
| -model_dir <local-directory> | 指定存储模型的本地路径. | e.g. `-model-dir "./models/"` |
| -datasrouce <datasource-connection-url> | 配置连接的相关数据引擎. | e.g. `-datasource "mysql://root:root@localhost:3306"` |

详细内容可以参考[Run SQLFlow REPL](https://sql-machine-learning.github.io/sqlflow/doc/run/repl/)


# 10.3 将分析模型固化到 SQLFlow 中的流程

看完了SQLFlow如何配置, 想必有些读者已经跃跃欲试想要为SQLFlow贡献模型了. 本章节将详细介绍如何将本地实现的分析模型固化到SQLFlow中.

1. Fork分支

    从Github上SQLFlow官方仓库中Fork出分支到个人仓库下, clone至本地开发环境.
    
    ```bash
    git clone https://github.com/<Your Github ID>/models.git
    ```

2. 编写模型

    在`models`目录下用户可以进行自定义模型的开发工作, 这里以基于`keras`的`mydnnclassifier.py`为例.

    ```
    sqlflow_models
       |- dnnclassifier.py
       |- mydnnclassifier.py
    ```

    `mydnnclassifier.py`代码样例.

    ```python
    import tensorflow as tf
        
    class MyDNNClassifier(tf.keras.Model):
       def __init__(self, feature_columns, hidden_units=[10,10], n_classes=2):
           ...
           ...
    ```

    SQLFlow会将从数据引擎中传入的数据根据具体类型和特征转换为TensorFlow中的对应的特征列(feature column)从而方便在模型中进行使用.

    特征列是TensorFlow为了方便用户进行特征工程而提供的数据概念封装. 以分桶列(bucketized_column)为例, 用户可以根据参数`boundaries`来将传入的`source_column`数据进行分桶, 从而实现连续数据的离散化处理.

    常用的特征列还包括数值列(numeric column), 分类词汇列(categorical vocabulary columns), 嵌入列(embedding column)等等.

    TensorFlow在官网教程中提供了对应的简介 [https://tensorflow.google.cn/tutorials/structured_data/feature_columns](https://tensorflow.google.cn/tutorials/structured_data/feature_columns).

    更详细的介绍可以参考API页面 [https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/feature_column](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/feature_column).

    其他参数对应`TRAIN`语句中传入的参数, 一般在这里定义一些模型相关的超参数和参数, 例如模型的层数, 具体每层的隐藏节点数等. 也可以这里通过声明参数的默认值来减少`TRAIN`语句中的输入.

    在使用自定义模型时, 如果`TRAIN`语句中没有传入对应的参数值, 而这里参数也没有默认值的话, 则会抛出异常终止训练.

    具体的模型结构和构建过程需要根据用户自身的需求决定. 建议用户在第一次自定义模型时, 借鉴仓库中已有的模型进行构建, 以避免不必要的误解.

3. 加载模型

    在`sqlflow_models/__init__.py`文件中导入自定义模型.

    ```bash
    from .mydnnclassifier import MyDNNClassifier
    ```

4. 单元测试

    使用`unittest`来进行自定义模型的单元测试工作, 相关测试用例可以补充在`sqlflow_models/tests`路径下.
    
    ```python
   from sqlflow_models import MyDNNClassifier
   from tests.base import BaseTestCases
    
   import tensorflow as tf
   import unittest
    
   class TestMyDNNClassifier(BaseTestCases.BaseTest):
       def setUp(self):
            self.features = {...}
            self.label = [...]
            feature_columns = [...]
            self.model = MyDNNClassifier(feature_columns=feature_columns)
    
    if __name__ == '__main__':
        unittest.main()
    ```

5. 集成测试

    进入`models`路径后启动对应`sqlflow/sqlflow`镜像对应container, 利用参数`-v`将当前目录挂载至容器中`/models`路径下, 运行端口8888.
    
    ```bash
    cd models/
    docker run --rm -it -v $PWD:/models -p 8888:8888 sqlflow/sqlflow
    ```
    
    利用`exec`命令在容器中更新`models`模块, 如果是非镜像用户则可以通过`pip -install -U models`在环境中进行`models`模块的更新.
    
    ```bash
    docker exec -it <container-id> pip install -U models
    ```
    
    通过在训练语句中指定模型为自定义模型进行测试. 这里以`iris`数据集为例进行自定义模型`MyDNNClassifier`的测试.

    ```sql
    SELECT * from iris.train
    TRAIN sqlflow_models.MyDNNClassifier
    WITH n_classes = 3, hidden_units = [10, 20]
    COLUMN sepal_length, sepal_width, petal_length, petal_width
    LABEL class
    INTO sqlflow_models.my_dnn_model;
    ```

    集成测试需要检查命令是否执行正常, 模型效果是否符合预期以及结果是否正确写入和读取. 如果以上目标皆达成, 恭喜, 你已经完成了在SQLFlow上的模型开发工作.



7. 贡献模型

    恭喜, 你已经完成了自定义的模型构建. 接下来, 你需要创建一个`pull request`并邀请其他开发者进行审查. 
    
    如果社区内的其他开发者通过了你的拉取请求, 那么你就可以将你的模型合并进官方代码仓库适合的分支中(目前是`develop`分支). 
    
    `travis-ci`每晚会自动构建对应的最新镜像并推送到`Docker Hub`中附带`nightly`标签. 第二天你就可以在最新镜像中使用自己的模型.

# 10.4 总结

至此, 你已经学会了如何使用SQLFlow进行模型训练和预测, 以及独立搭建SQLFlow的使用环境. 但SQLFlow的能力还远不止与此. 目前SQLFlow已经支持通过Kubernetes实现更加高效的工业化部署, 同时借力于Google提供的CloudSQL服务和Kubernetes Engine, 用户同样可以在Google Cloud平台上部署和使用SQLFlow.

如果你想了解SQLFlow是如何解析请求, 如果提交任务等更多设计上的细节, 请参考官网中的Design章节[https://sql-machine-learning.github.io/doc_index/sqlflow_designs](https://sql-machine-learning.github.io/doc_index/sqlflow_designs), 同样也欢迎用户在Github上踊跃提出自己的问题.

作为一个开源不到1年的新项目, SQLFlow仍在继续努力, 例如对接更多的框架和引擎, 支持更多的平台和系统, 开发和应用更多的模型. 而整个社区的建设与每一个使用者, 开发者和维护者所作出的努力密不可分. 非常希望阅读本书的你也能参与进来, 为SQLFlow的茁壮成长贡献自己的一份力量!

SQLFlow on GitHub : [https://github.com/sql-machine-learning/sqlflow](https://github.com/sql-machine-learning/sqlflow)

SQLFlow Homepage : [https://sqlflow.org](https://sqlflow.org)