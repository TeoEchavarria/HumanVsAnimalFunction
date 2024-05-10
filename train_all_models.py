from model import model_train
from predict import predict_outcome
from data_3 import dfs

for key, values in dfs.items():
    print(key, model_train(values, key, '18-34', 'male'))