#coding:utf-8
import cv2
import os
import math

# 把标注数据改成模型需要的格式
"""﻿
 120,49,546,49,546,157,120,157,PROPER  (xmin, ymin, xmax, ymax, text)
 121,163,549,163,549,327,121,327,FOOD
 122,329,546,329,546,434,122,434,PRONTO
 to: cls(not used), x/image_x (中心点x/图片宽),y/image_y(中心点y/图片长),h,w,angle,text
 0 0.5203125 0.21458333333333332 0.5325 0.135 0 PROPER
 0 0.5234375 0.5104166666666666 0.535 0.205 0 FOOD
 0 0.521875 0.7947916666666667 0.53 0.13125 0 PRONTO
"""

def is_chinese(name):
	for ch in name:
		if ord(ch) > 0x4e00 and ord(ch) < 0x9fff:
			return True
	return False

def get_gt_xywh(file_name):
	file_name = file_name.replace("Challenge2_Test_Task12_Images\\","Challenge2_Test_Task1_GT\\gt_")
	print(file_name)
	# 从txt中读取gt框, x y h w
	gt = []
	with open(file_name, "r", encoding="utf-8") as f:
		for line in f:
			xmin, ymin, xmax, ymax = line.split(",")[:4]
			text = ",".join(line.split(",")[4:])
			xmin, ymin, xmax, ymax = int(xmin) ,int(ymin), int(xmax), int(ymax)
			x1, y1, x2, y2, x3, y3, x4, y4  = xmin , ymin, xmax, ymin, xmax ,ymax, xmin , ymax
			text = text.replace("\"", "")
			gt.append([x1, y1, x2, y2, x3, y3, x4, y4, 0, text])
	return gt

def get_gt(file_name):
	#从txt中读取gt框, 四个点坐标
	gt = []
	with open(file_name,"r",encoding="utf-8") as f:
		for line in f:
			x1, y1, x2, y2, x3, y3, x4, y4, label, text = line.split(",")
			text = text.replace("\"","")
			gt.append([int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4), label, text])
	return gt

def walk_dir(folder_path):
	train_list = folder_path+"\\train.txt" # 训练文件列表
	open(train_list, "w", encoding="utf-8")

	paths = os.listdir(folder_path)
	for path in paths:
		if not path.endswith("jpg"):
			continue
		with open(train_list, "a+", encoding="utf-8") as f_train:
			f_train.write(path + "\n")
		pppp = os.path.join(folder_path, path)
		im = cv2.imread(pppp)
		image_width = im.shape[1]
		image_height = im.shape[0]
		norm = 1.0 / math.hypot(image_height, image_width)
		gt = get_gt(pppp.replace("jpg","txt"))

		result_path = pppp.replace("try","try_result").replace("jpg","txt")
		open(result_path, "w", encoding="utf-8")
		for rect in gt:
			x1, y1, x2, y2, x3, y3, x4, y4, cls, text = rect

			width = math.hypot(x2 - x1, y2 - y1)  # 欧几里德范数-勾股定理
			height = math.hypot(x2 - x3, y2 - y3)
			x = (x1 + x2 + x3 + x4) / 4.0
			y = (y1 + y2 + y3 + y4) / 4.0

			angle = math.atan2((y2 - y1), (x2 - x1))  # 角度
			conv = [x / image_width, y / image_height, width * norm, height * norm, angle]

			with open(result_path,"a+",encoding="utf-8") as annfile:
				annfile.write(str(0) + " " + " ".join([str(a) for a in conv]) + " " + rect[9] )

if __name__ == "__main__":
	folder_path = "E:\\icdar_2017\\try"
	# folder_path = "E:\\icdar2013\\Challenge2_Test_Task12_Images"
	walk_dir(folder_path)