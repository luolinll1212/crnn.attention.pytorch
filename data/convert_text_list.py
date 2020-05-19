import os
import sys
from PIL import Image

# with open('char_std_5990.txt') as fd:
#     cvt_lines = fd.readlines()
#
# cvt_dict = {}
# for i, line in enumerate(cvt_lines):
#     key = i
#     value = line.strip()
#     cvt_dict[key] = value

def check_file(path):
    if os.path.exists(path):
        os.remove(path)

def check_img(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        return True
    except IOError:
        return False

def gen_label_txt(root, target_list_txt, txt_files):
    # 读取原标签文件
    with open(os.path.join(root, txt_files), "r", encoding="utf-8") as fd:
        lines = fd.readlines()
    # 生成新标签文件
    count = 0
    with open(target_list_txt, "w+", encoding="utf-8") as file:
        for line in lines:
            lines = line.strip().split(' ')
            img_name = lines[0]

            img_path = os.path.join(path, "images", img_name)
            label = lines[1]

            if check_img(img_path):
                file.write("{0} {1}\n".format(img_path, label))
            else:
                print(img_path, " ", label)

            if count % 200000 == 0:
                print(count)

            count += 1


if __name__ == "__main__":

    path = "/home/rose/data/OCR/Synthetic_Chinese_String_Dataset"



    # # 输出训练标签文件
    train_list_txt = "train_list.txt"
    # check_file(train_list_txt)
    # gen_label_txt(root=path, target_list_txt=train_list_txt, txt_files="train.txt")
    #
    # # 输出测试标签文件
    # test_list_txt = "test_list.txt"
    # check_file(test_list_txt)
    # gen_label_txt(root=path, target_list_txt=test_list_txt, txt_files="test.txt")

    # 提取标签文件
    with open(train_list_txt, "r", encoding='utf-8') as file:
        # for line in file.readlines():
        #     line = line.strip().split(' ')
        #     img_path = line[0]
        #     label = line[1]
        #     print(img_path)
        #     print(label)
        #     exit()
        print(len(file.readlines()))
