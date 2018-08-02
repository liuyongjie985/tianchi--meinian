main.py 相关说明:
    本项目已经按照要求, 执行main.py即可一次性将结果csv保存到submit目录下,全程均无任何问题.
    但是,由于其中包含大量的特征工程操作,将经历多个执行步骤,所以耗费的执行时间较长(大大长于模型的训练和预测步骤).
    为此,我们在main.py的main函数中设置了变量isFast, 如果令isFast=0,则代码正常按顺序执行每个步骤,最后得出结
    果,如果令isFast=1(默认值,推荐使用),则代码将跳过一些步骤的处理,改为直接读取该步骤完成后所生成的中间文件,从
    而大大提高出结果的速度.
    下面将具体说明执行较慢的步骤,以及设置isFast的情况:
        1. data re-build 步骤对训练数据和测试数据进行清洗组合等操作,最终生成train_set.csv和test_set.csv
        两个文件.由于文件较大,所以无法放入项目中,也就无法通过isFast使其加快.还恳请您等待程序执行完毕.
        2. data split 步骤对训练数据和测试数据进行筛选,目的是选出数值型数据列,分类型数据列,以及文本型数据列,
        为了保证分类准确,我们对数据进行了遍历,所以导致程序运行缓慢.我们将分组结果保存到了code/files/目录下的
        'categoricalData_tableID','NumericData_tableID'和'StringData_tableID'文件中.如果设置main.py
        的isFast=1, 将直接读取该文件而跳过处理步骤.
        3. data wash 步骤将NumericData_tableID中的tableID从train_set.csv和test_set.csv中抽出,清洗并存
        入新文件'NumericData_clean_train.csv'和'NumericData_clean_test.csv'中,执行速度较慢.我们将两个生
        成好的文件保存到了code/files/目录下,如果设置main.py
        的isFast=1, 将直接读取该文件而跳过处理步骤.