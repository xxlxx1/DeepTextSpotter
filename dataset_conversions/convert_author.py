#coding:utf-8
import cv2
import os
import math

# 把标注数据改成模型需要的格式
"""﻿
 120,49,546,49,546,157,120,157,PROPER
 121,163,549,163,549,327,121,327,FOOD
 122,329,546,329,546,434,122,434,PRONTO
 to: cls(not used), x/image_x,y/image_y,h,w,angle,text
 0 0.5203125 0.21458333333333332 0.5325 0.135 0 PROPER
 0 0.5234375 0.5104166666666666 0.535 0.205 0 FOOD
 0 0.521875 0.7947916666666667 0.53 0.13125 0 PRONTO
"""

def is_chinese(name):
	for ch in name:
		if ord(ch) > 0x4e00 and ord(ch) < 0x9fff:
			return True
	return False

def get_gt(file_name):
	#从txt中读取gt框
	gt = []
	with open(file_name,"r",encoding="utf-8") as f:
		for line in f:
			x1, y1, x2, y2, x3, y3, x4, y4, label, text = line.split(",")
			text = text.replace("\"","")
			gt.append([int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4), label, text])
	return gt

def walk_dir(folder_path):
	train_list = "E:\\icdar_2017\\train.txt" # 训练文件列表
	open(train_list, "w", encoding="utf-8")

	paths = os.listdir(folder_path)
	for path in paths:
		if not path.endswith("jpg"):
			continue
		with open(train_list, "a+", encoding="utf-8") as f_train:
			f_train.write(path + "\n")
		pppp = os.path.join(folder_path, path)
		im = cv2.imread(pppp)
		gt = get_gt(pppp.replace("jpg","txt"))
		dw = 1. / im.shape[1]
		dh = 1. / im.shape[0]

		result_path = pppp.replace("try","try_result").replace("jpg","txt")
		for rect in gt:
			width = math.hypot(rect[2] - rect[0], rect[3] - rect[1])
			height = math.hypot(rect[2] - rect[4], rect[3] - rect[5])
			x = (rect[0] + rect[2] + rect[4] + rect[6]) / 4.0
			y = (rect[1] + rect[3] + rect[5] + rect[7]) / 4.0

			angle = math.atan2((rect[3] - rect[1]), (rect[2] - rect[0]))

			norm = math.sqrt(im.shape[1] * im.shape[1] + im.shape[0] * im.shape[0])
			norm = 1.0 / norm

			conv = [x, y, width, height, angle]
			conv[0] *= dw
			conv[1] *= dh
			conv[2] *= norm
			conv[3] *= norm

			cls_id = 0
			if is_chinese(rect[9]):
				cls_id = 1
			else:
				print(rect[9])

			with open(result_path,"a+",encoding="utf-8") as annfile:
				annfile.write(str(cls_id) + " " + " ".join([str(a) for a in conv]) + " " + rect[9] )

if __name__ == "__main__":
	folder_path = "E:\\icdar_2017\\try"
	walk_dir(folder_path)