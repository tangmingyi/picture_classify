import xgboost as xgb
from xgboost import  plot_importance
from matplotlib import pyplot as plt
import numpy as np
dtrain = xgb.DMatrix("temp.svm.txt",missing=0,weight=None)
# data = np.array([1,2,3]).reshape([3,1])
# label = np.array([1,2,3])
# dtest = xgb.DMatrix(data,label)
evalist = (dtrain,"tmp_test")
dtrain.save_binary("train_temp.buffer")
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 10,               # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 12,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
}
# plst = list(params.items())
num_round = 1
model = xgb.train(params=params,dtrain=dtrain,num_boost_round=num_round,evals=[evalist])
res = model.predict(dtrain)
plot_importance(model)
plt.show()