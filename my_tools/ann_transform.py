import os
import glob
import math


train_data_dir = 'E:\\137\\学业\\研究生\\课堂\\第二学期\\模式识别实验\\data\\MunichDatasetVehicleDetection-2015-old\\Train'
output_dota_ann_dir = 'dota_ann_files'
CONVERT_TO_INTEGER = False  # convert to integer if necessary


if __name__ == '__main__':
    img_paths = glob.glob(os.path.join(train_data_dir, '*.JPG'))
    if not os.path.exists(output_dota_ann_dir):
        os.makedirs(output_dota_ann_dir)

    categories = []
    label_num = 0
    for img_path in img_paths:
        img_prefix = img_path.split('.JPG')[0]
        img_name = img_prefix.split('\\')[-1]
        ann_paths = glob.glob(img_prefix + '*.samp')

        with open(os.path.join(output_dota_ann_dir, img_name + '.txt'), 'w') as dota_ann_file:
            dota_ann_file.write("'imagesource':DLR-MVDA\n")
            dota_ann_file.write("'gsd':null\n")

            for ann_path in ann_paths:
                with open(ann_path, 'r') as ann_file:
                    category = ann_path.split(
                        'Tunnel_')[-1].split('_', maxsplit=1)[-1].split('.')[0]
                    if category not in categories:
                        categories.append(category)

                    lines = ann_file.readlines()
                    for line in lines:
                        if line[0] == '#' or line[0] == '@':
                            continue

                        _, _, cx, cy, width, height, angle = map(
                            float, line.strip().split(' '))

                        if width == 0 or height == 0:
                            continue

                        angle = math.radians(angle)
                        half_diagonal_len = math.sqrt(width**2 + height**2)
                        angle_diff = math.atan(height/width)
                        angle1 = angle + angle_diff
                        angle2 = angle - angle_diff

                        x1 = cx + half_diagonal_len*math.cos(angle1)
                        x2 = cx + half_diagonal_len*math.cos(angle2)
                        x3 = cx - half_diagonal_len*math.cos(angle1)
                        x4 = cx - half_diagonal_len*math.cos(angle2)

                        y1 = cy + half_diagonal_len*math.sin(angle1)
                        y2 = cy + half_diagonal_len*math.sin(angle2)
                        y3 = cy - half_diagonal_len*math.sin(angle1)
                        y4 = cy - half_diagonal_len*math.sin(angle2)

                        if CONVERT_TO_INTEGER:
                            x1, x2, x3, x4, y1, y2, y3, y4 = map(
                                round, [x1, x2, x3, x4, y1, y2, y3, y4])

                        x1, x2, x3, x4, y1, y2, y3, y4 = map(
                            str, [x1, x2, x3, x4, y1, y2, y3, y4])

                        sep = ' '
                        seq = [x1, y1, x2, y2, x3, y3, x4, y4, category, '0']
                        line_to_write = sep.join(seq) + '\n'
                        dota_ann_file.write(line_to_write)
                        label_num = label_num + 1

    print(categories)
    print(label_num)
