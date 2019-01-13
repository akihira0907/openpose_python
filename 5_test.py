#coding:utf-8
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import re
import csv
import matplotlib.pyplot as plt
from sys import platform
import numpy as np
from time import sleep
import time

# ----------class points----------
class Point():
  def __init__(self, x, y):
    self.x = x
    self.y = y

# ----------count the number of files----------
def count_files():
  directory = os.path.expanduser('~') + "/opencv_test/images"
  files = os.listdir(directory)
  count = 0
  for file in files:
    index = re.search('.png', file)
    if index:
      count = count + 1
  return count

# ---------search neighborhood point's index----------
def search_neighborhood(pt0, pts):
  distances = np.repeat(0, pts.shape[0])
  for i, pt in enumerate(pts):
    distances[i] = np.linalg.norm(pt - pt0)
  return distances.argmin()

# ---------get new person's coordinate from 1 to 2----------
def get_coordinate_1(last_coordinates, coordinates):
  for last_c in last_coordinates:
    index = search_neighborhood(last_c, coordinates)
  return coordinates[index - 1]

# ---------get new person's coordinate from 0 to 1----------
def get_coordinate_0(coordinates):
  return coordinates[0]

# ---------関数:csv書き込み----------
def write_csv(lst, path):
  """
  csvに書き出す

  Parameters
  ----------
  lst: list
    書き込む座標のリスト
  path: string
    書き込むcsvファイル名
  """
  # if not os.path.isfile(path): # ファイルが存在しない場合wで開く
  #   with open(path, 'w') as f:
  #     writer = csv.writer(f, lineterminator='\n')
  #     writer.writerow(lst)
  # else: # ファイルが存在する場合aで開く
  #   with open(path, 'a') as f:
  #     writer = csv.writer(f, lineterminator='\n')
  #     writer.writerow(lst)

  #ファイルの存在の有無に関わらず上書き
  with open(path, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(lst)

# ---------match coordinates---------
def match_coodinates(last_coordinates, raw_coordinates):
  """
  タグ付けされてない座標と前フレームの座標から
  現在フレームのタグ付けされた座標を取得

  Parameters
  ----------
  last_coordinates: dict
    前フレームのタグ付けされた座標
  raw_coordinates: list (int)
    タグ付けされてない今フレームの座標

  Returns
  -------
  coordinates: dict
    タグ付けされた今フレームの座標
  new_coordinates: dict
    タグ付けされた新しい人物の座標
  """
  last_coordinates_items = last_coordinates.items() # 辞書をタプルのリストに変換
  new_person_flags = np.ones(len(raw_coordinates), dtype=np.bool) # 新しい座標か否かの真偽値の配列(サイズは今フレームの座標の数)
  distance_matrix = np.zeros([len(last_coordinates), len(raw_coordinates)]) # 距離を格納する二次元配列
  coordinates = dict() # 返戻値
  new_coordinates = dict() # 返戻値

  # 距離の二次元配列を埋めていく
  for i, (_, last_c) in enumerate(last_coordinates_items): # 前フレームの座標の数だけループ
    for j, raw_c in enumerate(raw_coordinates): # 今フレームの座標の数だけループ
      distance_matrix[i][j] = np.sqrt((last_c[0]-raw_c[0])**2 + (last_c[1]-raw_c[1])**2) #2点間の距離を計算
  
  # 距離が最小のものから処理を行っていく
  for count in range(min(distance_matrix.shape)): # 前と今フレームの少ない方の座標の数だけループ
    i, j = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape) # 距離が最小の組み合わせを取得
    coordinates[last_coordinates_items[i][0]] = raw_coordinates[j] # 対応するIDをキーとして座標を格納
    distance_matrix[i, :] = np.Inf # 対応してる距離を無限に
    distance_matrix[:, j] = np.Inf # 対応してる距離を無限に
    new_person_flags[j] = False # 対応してる真偽値を偽に

  # 新しい人物の座標について処理
  for i, flag in enumerate(new_person_flags): # 真偽値についてループ
    if flag == True:
      key = '{}_{}'.format(time.time(), i) # 新しいIDを生成
      coordinates[key] = raw_coordinates[i] # 新しい座標を新しいキーで登録
      new_coordinates[key] = raw_coordinates[i] # 新しい座標だけのリストにも登録

  return coordinates, new_coordinates

# ----------パラメータのセット----------
# Remember to add your installation path here
# Option a
dir_path = os.path.dirname(os.path.realpath(__file__))
if platform == "win32": sys.path.append(dir_path + '/../../python/openpose/');
else: sys.path.append('../../python');
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/../../../models/"
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

# ----------print keypoints infinity and write csv----------
current_num = 0 # 現在の人数
last_num = 0 # 1フレーム前の人数
coordinates = dict() # 現在のフレームのタグ付けされた座標の辞書
last_coordinates = dict() # 前フレームのタグ付けされた座標の辞書
i = 0 # index for loop
not_found_count = 0 # counter of not found
path = 'output.csv' # 出力先csvファイル名

# 出力ファイルの削除(初期化)
if os.path.isfile(path):
  os.remove(path)

cap = cv2.VideoCapture(0) # カメラを作成
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # カメラの幅
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # カメラの高さ

# 無限ループ
while True:
  ret, img = cap.read()
  if ret:
    keypoints, output_image = openpose.forward(img, True)
    i = i + 1
    not_found_count = 0 # reset counter
  else:
    print("serarching images...")
    sleep(0.5)
    not_found_count = not_found_count + 1
    if not_found_count > 10: sys.exit(0) # not found images so finish
    continue # back to loop
  
  # show image
  cv2.imshow("output", output_image)
  # print("Frame" + str(i))

  # find new person
  new_person_flag = False
  last_num = current_num
  current_num = keypoints.shape[0]
  if current_num > last_num:
    new_person_flag = True
    print("find a new person!")
    print("detect " + str(last_num) + " -> " + str(current_num) + " persons.")

  # 座標を計算、格納
  last_coordinates = coordinates # 前フレームの座標を格納
  raw_coordinates = [] # タグ付けされてない座標を格納するリスト
  if keypoints.shape[0] != 0: # 人数がゼロ人じゃなければ計算
    for j in range(0, keypoints.shape[0]): # 人の数だけループを回す
      x = keypoints[j, :, 0]
      y = keypoints[j, :, 1]
      size = 0
      # for k in range(0, keypoints.shape[1]): # 座標の数だけループを回す
      #   if not (keypoints[0, :, 0][k]==0 and keypoints[0, :, 1][k]==0): # 座標のxyが非ゼロならカウント
      #     size = size + 1
      # # 座標を有効な数で割って平均を算出
      # x_mean = np.sum(x) / size
      # y_mean = np.sum(y) / size
      x_mean = x[0] # 鼻の座標（０）だけ取ってみる
      y_mean = y[0]
      if new_person_flag: # 人数が増加していれば座標を表示
        print("x(" + str(j) + "): " + str(x_mean))
        print("y(" + str(j) + "): " + str(y_mean))
      raw_coordinates.append([x_mean, y_mean]) # 新しい座標をリストに追加
  # タグ付けされた座標の辞書と新しい座標の辞書を取得
  coordinates, new_coordinates = match_coodinates(last_coordinates, raw_coordinates)

  # 新しい人物の座標を取得、csvに書き込み
  # if new_person_flag:
  #   if current_num == 1 and last_num == 0:
  #     new_person_coordinate = get_coordinate_0(np.array(coordinates)).tolist()
  #     print("new persons's coordinate: ")
  #     print(new_person_coordinate)
  #     write_csv(new_person_coordinate, path) # csvに出力
  #   elif current_num == 2 and last_num == 1:
  #     new_person_coordinate = get_coordinate_1(np.array(last_coordinates), np.array(coordinates)).tolist()
  #     print("new persons's coordinate: ")
  #     print(new_person_coordinate)
  #     write_csv(new_person_coordinate, path) # csvに出力
  #   else:
  #     print("unsupport number of people") 
  #   print("--------------------")

  # 増えた人数が1人ならCSVに書き込み
  if current_num - last_num == 1:
    print("new persons's coordinate: ")
    print(new_coordinates.values()[0])
    write_csv(new_coordinates.values()[0], path) # csvに出力
    print("--------------------")

  cv2.waitKey(15)


