from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from utils import mix_seed, process
from get_data import generate_multimodal_data
from mmae import mmae
import argparse
import warnings
from early_stopping import EarlyStopping
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Process some parameters.')
# 添加参数
parser.add_argument('--epoch', type=int, default=1, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--f1', type=int, default=64, help='F1 value')
parser.add_argument('--dropout_rate', type=float, default=0.10, help='Dropout rate')
parser.add_argument('--path', type=str, default="nf10k.xlsx", help='Path to the input file')
parser.add_argument('--target', type=str, nargs='+', default=['permeance', 'rejection'], help='Target columns')
parser.add_argument('--seed', type=int, default=123456, help='Random seed')
parser.add_argument('--method', type=str, default='z', help='Method to use')
parser.add_argument('--is_df', type=bool, default=True, help='Whether to use DataFrame (scikit-rtdl)')
parser.add_argument('--factor', type=float, default=1.0, help='Factor value')
parser.add_argument('--delete_smile', type=bool, default=False, help='Whether to delete smile')
args = parser.parse_args()
# x_train, x_val, x_test, y_train, y_val, y_test, y_transformer
'''
In the hyperparameter tuning stage, this paper divides the dataset into 
training set, validation set and test set according to the ratio of 7:1:2, 
and then firstly determines the approximate range of the optimal parameters 
through the random parameterization method, and finally determines the optimal 
parameters through grid search. All models are subjected to 300 sets of 
hyper-parameter optimization experiments in the validation set, and the random 
seed adopts a fixed value of 123456 to ensure the fairness as much as possible. 
After completing the tuning stage, the training set and validation set of the 
tuning stage are then merged into a training set, and the models are trained 
under the optimal parameters, and compared on the test set for a comprehensive 
evaluation of the performance of each model.
'''
x_train, x_test, y_train, y_test = generate_multimodal_data(path=args.path,
                                                            method=args.method,
                                                            is_df=args.is_df,
                                                            factor=args.factor,
                                                            target=args.target,
                                                            delete_smile=args.delete_smile,
                                                            seed=args.seed)
mix_seed(args.seed)

# print(x_train.columns)
train_list = process(x=x_train, y=y_train)
# val_list = process(x=x_val, y=y_val)
test_list = process(x=x_test, y=y_test)
train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
train_loader1 = DataLoader(train_list, batch_size=len(train_list), shuffle=True)
# val_loader = DataLoader(val_list, batch_size=len(val_list), shuffle=False)
test_loader = DataLoader(test_list, batch_size=len(test_list), shuffle=False)
mmae = mmae(f1=args.f1, dropout_rate=args.dropout_rate)
optimizer = torch.optim.Adam(mmae.parameters(), lr=args.lr)  # lr
loss_func = torch.nn.MSELoss()
early_stopping = EarlyStopping(patience=50)

def train(model=mmae, optimizer=optimizer, criterion=loss_func, epoch=args.epoch):
    for _ in range(epoch):
        for data in train_loader:
            # print(data.tb.shape)
            train_pre, reconstructed, attention_weights, h = model(data)
            train_loss = criterion(train_pre, data.y) + criterion(reconstructed, h)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # for data_val in train_loader:
        #     prediction_val, _, _, _ = model(data_val)
        #     r2_val_1 = r2_score(data_val.y.cpu().detach().numpy()[:, 0], prediction_val.cpu().detach().numpy()[:, 0])
        #     r2_val_2 = r2_score(data_val.y.cpu().detach().numpy()[:, 1], prediction_val.cpu().detach().numpy()[:, 1])
        #     val = (r2_val_1+r2_val_2)/2
        #     early_stopping(val)
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
    model.eval()  # 模型冻结
    # 输出最终的模型评估结果
    print("------------------------结果------------------------")
    # 计算训练集上的R²
    r2_train_1, r2_train_2, r2_test_1, r2_test_2 = 0, 0, 0, 0
    for i in train_loader1:
        prediction_train, _, _, _ = model(i)

        r2_train_1 = r2_score(i.y.cpu().detach().numpy()[:, 0], prediction_train.cpu().detach().numpy()[:, 0])
        r2_train_2 = r2_score(i.y.cpu().detach().numpy()[:, 1], prediction_train.cpu().detach().numpy()[:, 1])
        rmse_train1 = np.sqrt(
            mean_squared_error(i.y.cpu().detach().numpy()[:, 0], prediction_train.cpu().detach().numpy()[:, 0]))
        rmse_train2 = np.sqrt(
            mean_squared_error(i.y.cpu().detach().numpy()[:, 1], prediction_train.cpu().detach().numpy()[:, 1]))
        print(f'train-R2: RWP：{r2_train_1},RSP {r2_train_2}, train-rmse: RWP：{rmse_train1},RSP {rmse_train2}\n')
    # 计算测试集上的R²
    for j in test_loader:
        prediction_test, _, _, _ = model(j)

        r2_test_1 = r2_score(j.y.cpu().detach().numpy()[:, 0], prediction_test.cpu().detach().numpy()[:, 0])
        r2_test_2 = r2_score(j.y.cpu().detach().numpy()[:, 1], prediction_test.cpu().detach().numpy()[:, 1])
        rmse_test1 = np.sqrt(
            mean_squared_error(j.y.cpu().detach().numpy()[:, 0], prediction_test.cpu().detach().numpy()[:, 0]))
        rmse_test2 = np.sqrt(
            mean_squared_error(j.y.cpu().detach().numpy()[:, 1], prediction_test.cpu().detach().numpy()[:, 1]))
        print(f'test: RWP：{r2_test_1},RSP {r2_test_2}, test-rmse: RWP：{rmse_test1},RSP {rmse_test2}\n')
    print(r2_train_1, r2_train_2, r2_test_1, r2_test_2)
    sum_r2 = r2_test_1 + r2_test_2


import datetime

starttime = datetime.datetime.now()
train(model=mmae, optimizer=optimizer, criterion=loss_func)
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
