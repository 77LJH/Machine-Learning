import pandas as pd
import numpy as np
import copy
import bpnnUtil
from sklearn import datasets

class BpNN():
    def __init__(self,layer_dims_,learning_rate=0.1,seed=16,initializer='he',optimizer='gd'):
        self.layer_dims=layer_dims_ # layer_dims_ 是一个列表，指定神经网络每一层中神经元的数量，包括hidden layer和output layer
        self.learning_rate=learning_rate
        self.seed=seed # seed 是一个整数，用于初始化随机数生成器。
        self.initializer=initializer
        self.optimizer=optimizer

    def fit(self,X_,y_,num_epochs=100):
        m,n=X_.shape
        layer_dims_=copy.deepcopy(self.layer_dims) # 复制了神经网络的层维度 layer_dims_，以便稍后用于初始化网络参数。
        layer_dims_.insert(0,n) # 在层维度列表的开头插入 n，以确保输入层的神经元数等于特征数。

        if y_.ndim==1:
            y_=y_.reshape(-1,1) # 如果 y_ 是一维的，它会被改变形状，以确保它变成一个列向量

        # 这行代码断言神经网络的参数初始化方式必须是 'he' 或 'xavier' 中的一个。如果不是，会引发 AssertionError。
        assert self.initializer in ('he','xavier')
        if self.initializer=='he':
            self.parameters_=bpnnUtil.xavier_initializer(layer_dims_,self.seed)
        elif self.initializer=='xavier':
            self.parameters_=bpnnUtil.xavier_initializer(layer_dims_,self.seed)
        
        assert self.optimizer in ('gd', 'sgd', 'adam', 'momentum')
        if self.optimizer == 'gd':
            parameters_, costs = self.optimizer_gd(X_, y_, self.parameters_, num_epochs, self.learning_rate)
        elif self.optimizer == 'sgd':
            parameters_, costs = self.optimizer_sgd(X_, y_, self.parameters_, num_epochs, self.learning_rate, self.seed)
        elif self.optimizer == 'momentum':
            parameters_, costs = self.optimizer_sgd_monment(X_, y_, self.parameters_, beta=0.9, num_epochs=num_epochs,
                                                            learning_rate=self.learning_rate, seed=self.seed)
        elif self.optimizer == 'adam':
            parameters_, costs = self.optimizer_sgd_adam(X_, y_, self.parameters_, beta1=0.9, beta2=0.999, epsilon=1e-7,
                                                         num_epochs=num_epochs, learning_rate=self.learning_rate,
                                                         seed=self.seed)
        self.parameters_ = parameters_
        self.costs = costs
        
        return self

    def predict(self,X_):
        if not hasattr(self,'parameters_'): # 检查是否已经在模型对象 (self) 中定义了属性 parameters_
            raise Exception('you have to fit first before predict')

        a_last,_=self.forward_L_layer(X_,self.parameters_)
        """
        如果 a_last 的形状是 (m, 1)，其中 m 是样本数量，表示进行二元分类。这时，阈值为 0.5，大于等于 0.5 的输出被标记为类别 1，小于 0.5 的输出被标记为类别 0。
        如果 a_last 的形状不是 (m, 1)，则假定进行多类分类，并找到每个样本的预测类别。"""
        if a_last.shape[1]==1:
            predict_=np.zeros(a_last.shape)
            predict_[a_last>=0.5]=1
        else:
            predict_=np.argmax(a_last,axis=1)
        return predict_
    
    def compute_cost(self,y_hat_,y_):
        if y_.ndim==1: # 一维的目标数据通常用于二元分类问题
            y_=y_.reshape(-1,1)
        if y_.shape[1]==1:
            cost=bpnnUtil.cross_entry_sigmoid(y_hat_,y_)
        else:
            cost=bpnnUtil.cross_entry_softmax(y_hat_,y_)
        return cost

    def forward_one_layer(self,a_pre_,w_,b_,activation_):
        # 当执行相加操作时，Python将自动应用广播规则，使得形状不匹配的维度可以进行相加
        z_=np.dot(a_pre_,w_.T)+b_ # matrix1 和 matrix2 的最后一个维度（第二维度）都是大小为 N，所以它们可以相加
        assert activation_ in ('sigmoid','relu','softmax')

        if activation_=='sigmoid':
            a_=bpnnUtil.sigmoid(z_)
        elif activation_=='relu':
            a_=bpnnUtil.relu(z_)
        else:
            a_=bpnnUtil.softmax(z_)
        
        cache_=(a_pre_,w_,b_,z_) # 将向前传播过程中产生的数据保存下来，在向后传播过程计算梯度的时候要用上的。
        return a_,cache_
    
    def forward_L_layer(self,X_,parameters_):
        L_=int(len(parameters_)/2)
        caches=[]
        a_=X_
        for i in range(1,L_):
            w_=parameters_('W'+str(i))
            b_=parameters_('b'+str(i))
            a_pre_=a_
            a_,cache_=self.forward_one_layer(a_pre_,w_,b_,'relu')
            caches.append(cache_)
        
        w_last=parameters_['W'+str(L_)] # w_last 是神经网络输出层的权重矩阵
        b_last=parameters_['b'+str(L_)]

        if w_last.shape[0]==1: # 如果行数为1，表示神经网络用于二元分类，即只有一个输出单元，所以选择使用 Sigmoid 激活函数
            a_last,cache_=self.forward_one_layer(a_,w_last,b_last,'sigmoid')
        else:
            a_last,cache_=self.forward_one_layer(a_,w_last,b_last,'softmax')
        
        caches.append(cache_)
        return a_last,caches
    
    def backward_one_layer(self, da_, cache_, activation_):
        # 在activation_ 为'softmax'时， da_实际上输入是y_， 并不是
        (a_pre_, w_, b_, z_) = cache_
        m = da_.shape[0]

        assert activation_ in ('sigmoid', 'relu', 'softmax')

        if activation_ == 'sigmoid':
            dz_ = bpnnUtil.sigmoid_backward(da_, z_)
        elif activation_ == 'relu':
            dz_ = bpnnUtil.relu_backward(da_, z_)
        else:
            dz_ = bpnnUtil.softmax_backward(da_, z_)

        dw = np.dot(dz_.T, a_pre_) / m
        db = np.sum(dz_, axis=0, keepdims=True) / m
        da_pre = np.dot(dz_, w_)

        assert dw.shape == w_.shape
        assert db.shape == b_.shape
        assert da_pre.shape == a_pre_.shape

        return da_pre, dw, db
    
    def backward_L_layer(self,a_last,y_,caches):
        grads={}
        L=len(caches)

        if y_.ndim==1:
            y_=y_.reshape(-1,1)
        
        if y_.shape[1] == 1:  # 目标值只有一列表示为二分类
            da_last = -(y_ / a_last - (1 - y_) / (1 - a_last))
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(da_last, caches[L - 1], 'sigmoid')

        else:  # 经过one hot，表示为多分类

            # 在计算softmax的梯度时，可以直接用 dz = a - y可计算出交叉熵损失函数对z的偏导， 所以这里第一个参数输入直接为y_
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(y_, caches[L - 1], 'softmax')

        grads['da' + str(L)] = da_pre_L_1
        grads['dW' + str(L)] = dwL_
        grads['db' + str(L)] = dbL_

        for i in range(L-1,0,-1):
            da_pre_,dw,db=self.backward_one_layer(grads['da'+str(i+1)],caches[i-1],'relu')

            grads['da' + str(i)] = da_pre_
            grads['dW' + str(i)] = dw
            grads['db' + str(i)] = db

        return grads
    
    def optimizer_gd(self, X_, y_, parameters_, num_epochs, learning_rate):
        costs = []
        for i in range(num_epochs):
            a_last, caches = self.forward_L_layer(X_, parameters_)
            grads = self.backward_L_layer(a_last, y_, caches)

            parameters_ = bpnnUtil.update_parameters_with_gd(parameters_, grads, learning_rate)
            cost = self.compute_cost(a_last, y_)

            costs.append(cost)

        return parameters_, costs

    def optimizer_sgd(self, X_, y_, parameters_, num_epochs, learning_rate, seed):
        '''
        sgd中，更新参数步骤和gd是一致的，只不过在计算梯度的时候是用一个样本而已。
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]
        for _ in range(num_epochs):
            random_index = np.random.randint(0, m_)

            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_) # X_[[random_index], :]保持跟X同一纬度
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            parameters_ = bpnnUtil.update_parameters_with_sgd(parameters_, grads, learning_rate)

            a_last_cost, _ = self.forward_L_layer(X_, parameters_)

            cost = self.compute_cost(a_last_cost, y_)

            costs.append(cost)

        return parameters_, costs

    def optimizer_sgd_monment(self, X_, y_, parameters_, beta, num_epochs, learning_rate, seed):
        '''

        :param X_:
        :param y_:
        :param parameters_: 初始化的参数
        :param v_:          梯度的指数加权移动平均数
        :param beta:        冲量大小，
        :param num_epochs:
        :param learning_rate:
        :param seed:
        :return:
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]
        velcoity = bpnnUtil.initialize_velcoity(parameters_)
        for _ in range(num_epochs):
            random_index = np.random.randint(0, m_)

            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            parameters_, v_ = bpnnUtil.update_parameters_with_sgd_momentum(parameters_, grads, velcoity, beta,
                                                                           learning_rate)
            a_last_cost, _ = self.forward_L_layer(X_, parameters_)
            cost = self.compute_cost(a_last_cost, y_)
            costs.append(cost)

        return parameters_, costs

    def optimizer_sgd_adam(self, X_, y_, parameters_, beta1, beta2, epsilon, num_epochs, learning_rate, seed):
        '''

        :param X_:
        :param y_:
        :param parameters_: 初始化的参数
        :param v_:          梯度的指数加权移动平均数
        :param beta:        冲量大小，
        :param num_epochs:
        :param learning_rate:
        :param seed:
        :return:
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]
        velcoity, square_grad = bpnnUtil.initialize_adam(parameters_)
        for epoch in range(num_epochs):
            random_index = np.random.randint(0, m_)

            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            parameters_, velcoity, square_grad = bpnnUtil.update_parameters_with_sgd_adam(parameters_, grads, velcoity,
                                                                                          square_grad, epoch + 1,
                                                                                          learning_rate, beta1, beta2,
                                                                                          epsilon)
            a_last_cost, _ = self.forward_L_layer(X_, parameters_)
            cost = self.compute_cost(a_last_cost, y_)
            costs.append(cost)

        return parameters_, costs



if __name__=='__main__':
    iris=datasets.load_iris()
    X=pd.DataFrame(iris['data'],columns=iris['feature_names'])
    X=(X-np.mean(X,axis=0))/np.var(X,axis=0)

    y=pd.Series(iris['target_names'][iris['target']])
    y=pd.get_dummies(y)

    bp=BpNN([3,3],learning_rate=0.003,optimizer='adam')
    bp.fit(X.values,y.values,num_epochs=2000)

    bp1 = BpNN([3, 3], learning_rate=0.003, optimizer='sgd')
    bp1.fit(X.values, y.values, num_epochs=2000)
    
    


