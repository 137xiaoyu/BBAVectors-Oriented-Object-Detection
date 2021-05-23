import os
import glob


def run(data_dir):
    img_list = sorted(glob.glob(os.path.join(data_dir, 'images', '*.png')))
    train_list = img_list[:576]
    test_list = img_list[576:]
    with open(os.path.join(data_dir, 'trainval.txt'), 'w') as f:
        for img_path in train_list:
            img_name = img_path.split('\\')[-1].split('.')[0]
            f.write(img_name + '\n')
    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for img_path in test_list:
            img_name = img_path.split('\\')[-1].split('.')[0]
            f.write(img_name + '\n')     


if __name__ == '__main__':
    data_dir = 'D:/137/dataset/MunichDatasetVehicleDetection-2015-old/DOTA'
    run(data_dir)
