from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import Queue as Q
from math import sin, cos, sqrt, atan2, radians

DEBUG = True
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('sensor_ids_filename', 'data/coordinate.csv',
                    'File containing sensor ids separated by comma.')
flags.DEFINE_float('k', 2, 'The number of kNN neighbors which is a hypter parameter')
flags.DEFINE_string('output_pkl_filename', 'data/sensor_graph/adj_mat.pkl', 'Path of the output file.')


def get_distance(lat_1, lng_1, lat_2, lng_2):
    # radius of earth in km
    R = 6373.0

    dlng = radians(lat_2) - radians(lat_1)
    dlat = radians(lng_2) - radians(lng_1)

    x = dlat*cos((lat_1+lat_2)*0.5)

    d = R * sqrt(x**2 + dlng**2)
    if DEBUG:
        print (d)
    return d


def kNN(sensor_id_list, lat_list, long_list, id, k=2):
    kNN_PQ = Q.PriorityQueue(maxsize=k)

    nearest_distance = np.inf
    nearest_id = -1
    for _id in sensor_id_list:
        if _id == id:
            continue
        dis = -get_distance(lat_list[id-1], long_list[id-1], lat_list[_id-1], long_list[_id-1])
        if not kNN_PQ.full():
            kNN_PQ.put([dis, _id])
        else:
            temp_data = kNN_PQ.get()
            temp_dis = temp_data[0]
            temp_label = temp_data[1]
            if dis > temp_dis:
                temp_dis = dis
                temp_label = _id
            kNN_PQ.put([temp_dis, temp_label])
    return kNN_PQ


def get_adjacency_matrix(sensor_id_list, sensor_name, lat_list, long_list, k=2):
    """
    :sensor_id_list: list of sensor id
    :lat_list:       list of latitude of sensor
    :long_list:      list of longtitude of sensor
    """
    sensor_num = len(sensor_id_list)
    dist_mx = np.zeros((sensor_num, sensor_num), dtype=np.float32)
    sensor_name2id = {}
    # Construct the graph using kNN, add edge to k nearest neighors
    for idx, id in enumerate(sensor_id_list):
        if DEBUG:
            print (sensor_name[idx], id-1)
        sensor_name2id[sensor_name[idx]] = id-1
        knn_pq = kNN(sensor_id_list, lat_list, long_list, id, k)
        knn_list = knn_pq.queue
        for nei in knn_list:
            dist_mx[id - 1, nei[1] - 1] = 1
            dist_mx[id - 1][id - 1] = 1
    if DEBUG:
        print (dist_mx)
    return sensor_name2id, dist_mx


if __name__ == '__main__':
    sensor_info = pd.read_csv(FLAGS.sensor_ids_filename, dtype={'senor_id':'int', 'senor_name':'str', 'Lat':'float', 'Long':'float'})
    sensor_id_list = sensor_info['senor_id'].tolist()
    sensor_name = sensor_info['sensor_name'].tolist()
    lat_list = sensor_info['Lat'].tolist()
    long_list = sensor_info['Long'].tolist()
    if DEBUG:
        print(sensor_id_list, lat_list, long_list)
    sensor_name2id, adj_mx = get_adjacency_matrix(sensor_id_list, sensor_name, lat_list, long_list)
    with open(FLAGS.output_pkl_filename, 'w') as f:
        pickle.dump([sensor_name, sensor_name2id, adj_mx], f)
