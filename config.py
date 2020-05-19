# *_*coding:utf-8 *_*
import pickle

# 数据集参数
train_list = "./data/test_list.txt"
eval_list = "./data/test_list.txt"
num_workers = 4
batch_size = 36
img_height = 32
img_width = 280
alphabet = pickle.load(open("./data/alphabet.pkl", 'rb'))
num_classes = len(alphabet) + 2 # SOS_TOKEN 和 EOS_TOKEN

# 训练参数
num_epochs = 10
hidden_size = 256
learning_rate = 0.0001
random_sample = True
teaching_forcing_prob = 0.5
max_width = 71
dropout_p = 0.1
model = "output"
pretrained = ""
interval = 1000

# demo
img_path = "./images/20456343_4045240981.jpg"
encoder = "./output/pretrained/encoder_9.pth"
decoder = "./output/pretrained/decoder_9.pth"
use_gpu = True
