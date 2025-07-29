import math
from moduls.SSIM import calculate_ssim
from moduls.Violet_base import list_all_files, write_pkl_file, read_pkl_file, get_parent_and_grandparent_dir, read_json_file,write_json_file
import cv2
import random
from image_match.goldberg import ImageSignature
import io
import numpy as np
import sys
import os


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.append(parent_parent_dir)
except: pass

def Distance_weighting_function(distance_mm, distance_sm):

    return (0.4 * (math.exp(distance_mm) - 1)) + (0.6 * (math.exp(distance_sm) - 1))



def ssim_distence(img_or_path1, img_or_path2):
    return 1 - calculate_ssim(img_or_path1, img_or_path2)

class MM_SSIM_map_match():

    def __init__(self):
        self.G_data = []
        self.map = [] #存储每个节点的父节点在fa_node中的索引
        self.fa_node = [] #存所有父节点
        #self.map = [] 
        self.search_tree = []
        self.min_distance = float('inf')
        self.insert_fa_node = []
        self.gis = ImageSignature()
        self.mm_distance_function = self.gis.normalized_distance
        self.sm_distance_function = ssim_distence
        self.dwfunc = Distance_weighting_function
        #self.gis.normalized_distance
    def load_pkl(self, pkl_path):
        self.G_data = read_pkl_file(pkl_path)
        #self.map = [0 for _ in range(len(self.G_data))]
        self.map = [-1 for _ in range(len(self.G_data))]
        
    def process_img_2_pkl(self, json_path, pkl_path):

        json_data = read_json_file(json_path)

        img_path_lsit = []
        img_id_list = []
        for one_data in json_data:
            img_path_lsit.append(one_data['img'])
            img_id_list.append(one_data['id'])

        data = []
        for id, path in zip(img_id_list, img_path_lsit):
            print(path)

            tmp_sm_sign = self.cvt_2_mosaic_gary(path)
            tmp_mm_sign = self.gis.generate_signature(path)
            
            
            data.append({'id': id, 'mm_vector': tmp_mm_sign, 'sm_vector': tmp_sm_sign})
            #end_time = time.time()
            #print(f"Time taken to get base_name: {end_time - start_time} seconds")
        write_pkl_file(pkl_path, data)
        self.G_data = data
        self.map = [-1 for _ in range(len(self.G_data))]

    def save_fa_map(self, pkl_path):
        data = {'fa':self.fa_node, 'map':self.map}
        write_pkl_file(pkl_path, data)

    def load_fa_map(self, pkl_path):
        data = read_pkl_file(pkl_path)
        self.fa_node = data['fa']
        self.map = data['map']

    def save_search_tree(self, pkl_path):
        data = {'tree': self.search_tree, 'min':self.min_distance}
        write_pkl_file(pkl_path, data)

    def load_search_tree(self, pkl_path):

        data = read_pkl_file(pkl_path)
        self.search_tree = data['tree']
        self.min_distance = data['min']

    def kmeans_plus_centroids(self, m, points=None):
        """
        使用k-means++的思想，选择分布均匀的m个中枢节点
        """
        if points is None:
            points = self.G_data

        def distance(p1, p2):
            return self.dwfunc(
                self.mm_distance_function(p1['mm_vector'], p2['mm_vector']),
                self.sm_distance_function(p1['sm_vector'], p2['sm_vector'])
            )

        n = len(points)
        indices = []
        indices.append(random.randint(0, n - 1))  # 随机选择第一个中心

        distances = np.full(n, np.inf)
        selected = np.zeros(n, dtype=bool)  # 记录哪些点已经被选择过

        for _ in range(1, m):
            # 计算所有点到最近中枢节点的距离
            for i in range(n):
                if selected[i]:
                    continue  # 已选中的点跳过

                d = distance(points[i], points[indices[-1]])

                # 如果距离是 NaN 或 inf，跳过这些点
                if np.isnan(d) or np.isinf(d):
                    continue  # 跳过无效点

                # 如果距离为0（即点与点完全相同），则设置为一个非常小的值（1e-9）
                if d == 0:
                    d = 1e-9  # 避免出现零距离

                distances[i] = min(distances[i], d)

            # 使用距离来计算选择下一个中枢节点的概率
            total_distance = distances.sum()

            if total_distance == 0:
                print("⚠️ 距离全为0或非法，采用均匀随机")
                remaining = list(set(range(n)) - set(indices))  # 剩余未选的点
                next_index = random.choice(remaining)  # 随机选择一个点
            else:
                # 使用概率来选择下一个中枢节点
                probs = distances / total_distance

                # 确保概率数组没有 NaN 或 inf 值
                if np.isnan(probs).any() or np.isinf(probs).any():
                    print(f"⚠️ 概率中含有 NaN 或 inf，采用均匀随机")
                    remaining = list(set(range(n)) - set(indices))  # 剩余未选的点
                    next_index = random.choice(remaining)  # 随机选择一个点
                else:
                    next_index = np.random.choice(n, p=probs)  # 使用概率选取中枢节点

            indices.append(next_index)
            selected[next_index] = True  # 标记为已选中

        self.fa_node = indices
        return indices


    def greedy_furthest_points(self, m, points = None, ):
        #001
        """
        使用贪心算法寻找彼此距离尽可能远的 m 个点。

        Args:
           points (list): 所有节点的坐标列表.
            m (int): 需要选择的节点数量。
            distance_function:  距离计算函数

        Returns:
            list: 选择的 m 个节点的索引列表。
        """
        #distance_function = self.distance_function

        if points == None:
            points = self.G_data

        n = len(points)
        if m > n:
            print("Error: m must be less or equal than n")
            return None
        selected_indices = []
        # 1. 随机选择一个点
        first_index = random.randint(0, n - 1)
        selected_indices.append(first_index)

        # 2. 循环选择 m - 1 个点
        cnt = 1
        for _ in range(m - 1):
        #while self.min_distance > 0.4:
            print(f'正在寻找，第{cnt+1}个节点')
            cnt+=1
            max_min_distance = -1
            best_index = -1
            for i in range(n):
                if i in selected_indices:
                    continue
                min_distance = float('inf')
                for selected_index in selected_indices:
                    distance =  self.dwfunc(self.mm_distance_function(points[i]['mm_vector'], points[selected_index]['mm_vector']),
                                            self.sm_distance_function(points[i]['sm_vector'], points[selected_index]['sm_vector']))
                    min_distance = min(min_distance, distance) #当前节点到当前集合中头节点的最小距离
                if min_distance > max_min_distance: #如果当前节点到头节点之间的最小距离大于当前头节点到其他节点的距离，则更新头节点和头节点到其他节点的距离，筛选出和所有头节点都尽可能不相邻的点
                    max_min_distance = min_distance
                    best_index = i

            if self.min_distance > max_min_distance: #记录头节点之间最小距离
                self.min_distance = max_min_distance
                print(f'当前最小距离{self.min_distance}')

            selected_indices.append(best_index)
            self.map[best_index] = len(selected_indices) - 1
        
        self.fa_node = selected_indices
        return selected_indices
    
    def process_fa_node_map(self, ):
        #002
        
        #distance_function = self.distance_function
        for data_id, data in enumerate(self.G_data):

            min_distance = float('inf')
            best_index = -1
            
            for fa_id, index in enumerate(self.fa_node):
                distance =  self.dwfunc(self.mm_distance_function(data['mm_vector'], self.G_data[index]['mm_vector']),
                                        self.sm_distance_function(data['sm_vector'], self.G_data[index]['sm_vector']))
                #distance_function(data['vector'], self.G_data[index]['vector'])
                
                if distance < min_distance:
                    min_distance = distance
                    best_index = fa_id
            
            self.map[data_id] = best_index
            print(f'data_id:{data_id},父节点为{best_index}')
        return self.map
    
    def generate_search_tree(self):
        #003
        self.search_tree = []  # 确保搜索树是空的，在每次生成时重新初始化
        node_point_counts = []  # 用于保存每个中枢节点的点的数量

        for index in range(len(self.fa_node)):
            tmp_tree = []
            tmp_tree.append(self.G_data[self.fa_node[index]])  # 添加父节点

            print(f'正在生成第{index+1}个节点的搜索树,当前父节点为{self.fa_node[index]}')

            # 遍历所有数据点
            for data_id, data in enumerate(self.G_data):
                if self.map[data_id] == index and data_id != self.fa_node[index]:
                    tmp_tree.append(data)  # 将属于当前父节点的点加入树中

            # 添加搜索树到最终结果
            self.search_tree.append(tmp_tree)

            # 保存当前父节点的点的数量
            node_point_counts.append(len(tmp_tree))

    
        # 在结束后打印每个中枢节点包含的点的数量
        for i, count in enumerate(node_point_counts):
            print(f"父节点 {self.fa_node[i]} 包含 {count} 个点")
        # 计算并打印方差
        variance = np.var(node_point_counts)
        print(f"点的数量的方差为: {variance}")

        return self.search_tree

    def calculate_centroid_distances(self):
        """
        计算所有中枢节点之间的距离，并评估这些距离的分布是否均匀
        """

        def distance(p1, p2):
            return self.dwfunc(
                self.mm_distance_function(p1['mm_vector'], p2['mm_vector']),
                self.sm_distance_function(p1['sm_vector'], p2['sm_vector'])
            )
        
        # 获取所有中枢节点的坐标
        centroids = [self.G_data[idx] for idx in self.fa_node]
        
        distances = []

        # 计算所有中枢节点之间的距离
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = distance(centroids[i], centroids[j])  # 使用之前定义的距离函数
                distances.append(dist)

        # 计算距离的方差
        distance_variance = np.var(distances)
        distance_mean = np.mean(distances)
        distance_min = np.min(distances)
        distance_max = np.max(distances)

        # 输出结果
        print(f"中枢节点之间的平均距离: {distance_mean}")
        print(f"中枢节点之间的最小距离: {distance_min}")
        print(f"中枢节点之间的最大距离: {distance_max}")
        print(f"中枢节点之间的距离方差: {distance_variance}")

        # 可选：绘制直方图来查看距离分布
        # import matplotlib.pyplot as plt
        # plt.hist(distances, bins=20, edgecolor='black')
        # plt.title('中枢节点之间的距离分布')
        # plt.xlabel('距离')
        # plt.ylabel('频率')
        # plt.show()

        return distances, distance_variance
    def search_img(self, img_or_path, ):

        #idstance_function = self.distance_function
        min_fa_distance = float('inf')
        #gis = self.gis
        
        mm_sign = self.gis.generate_signature(img_or_path)
        sm_sign = self.cvt_2_mosaic_gary(img_or_path)
        tree_id = -1

        #start_time = time.time()
        for index, tree in enumerate(self.search_tree):
            fa_node = tree[0]
            distance =  self.dwfunc(self.mm_distance_function(mm_sign, fa_node['mm_vector']),
                                    self.sm_distance_function(sm_sign, fa_node['sm_vector']))
            #distance_function(img_sign, fa_node['vector'])
            if distance < min_fa_distance:
                
                min_fa_distance = distance
                tree_id = index

        #if min_fa_distance > self.min_distance:
            #return None, min_fa_distance

        min_distance = float('inf')
        img_id = None
        for index, data in enumerate(self.search_tree[tree_id]):
            distance =  self.dwfunc(self.mm_distance_function(mm_sign, data['mm_vector']),
                                    self.sm_distance_function(sm_sign, data['sm_vector']))
            #distance_function(img_sign, data['vector'])
            # if index == 134: 
            #     print(self.mm_distance_function(mm_sign, data['mm_vector']))
            #     print(self.sm_distance_function(sm_sign, data['sm_vector']))
            #     pass
            if distance < min_distance:
                #print(distance)
                min_distance = distance
                img_id = data['id']
        #end_time = time.time()
        #print(f"Search time: {end_time - start_time} seconds")
        return img_id, min_distance
          
    def search_img_topk(self, img_or_path, thoshold = 0.6):

        #sorted(tuple_list, key=lambda x: x[1])
        
        topk_list = []

        #distance_function = self.distance_function
        min_fa_distance = float('inf')
        #gis = self.gis
        mm_sign = self.gis.generate_signature(img_or_path)
        sm_sign = self.cvt_2_mosaic_gary(img_or_path)
        tree_id = -1
        for index, tree in enumerate(self.search_tree):
            fa_node = tree[0]
            distance =  self.dwfunc(self.mm_distance_function(mm_sign, fa_node['mm_vector']),
                                    self.sm_distance_function(sm_sign, fa_node['sm_vector']))
            #distance =  distance_function(img_sign, fa_node['vector'])
            if distance < min_fa_distance:
                
                min_fa_distance = distance
                tree_id = index

        #min_distance = float('inf')
        #img_id = None
        for index, data in enumerate(self.search_tree[tree_id]):
            distance =  self.dwfunc(self.mm_distance_function(mm_sign, data['mm_vector']),
                                    self.sm_distance_function(sm_sign, data['sm_vector']))
            if distance <= thoshold:
                #print(distance)
                min_distance = distance
                #img_id = data['id']
                topk_list.append((data['id'], distance))
                
        if topk_list != []:
            return sorted(topk_list, key=lambda x: x[1])  
        else:
            return []
    
    def insert_new(self, img_or_path = None, signature = None, img_id = 'insert_img', ):
        #不调用G_data和fa_node fa_map进行插入，直接加入搜索树中
        #distance_function = self.distance_function
        
        #img = base64_to_ndarray(img_base64)
        mm_sign = self.gis.generate_signature(img_or_path)
        sm_sign = self.cvt_2_mosaic_gary(img_or_path)


        self.G_data.append({'id': img_id, 'mm_vector': mm_sign, 'sm_vector': sm_sign})


        min_distance = float('inf')
        insert_id = -1
        for id, data_list in enumerate(self.search_tree):
            fa_id = data_list[0]['id']
            fa_mm_vector = data_list[0]['mm_vector']
            fa_sm_vector = data_list[0]['sm_vector']
            distance = self.dwfunc(self.mm_distance_function(mm_sign, fa_mm_vector),
                                   self.sm_distance_function(sm_sign, fa_sm_vector))
            #distance_function(signature, fa_vector)
            if distance < min_distance:
                min_distance = distance
                insert_id = id

        if min_distance > self.min_distance:
            self.search_tree.append[[{'id': img_id, 'mm_vector': mm_sign, 'sm_vector': sm_sign}]]
        else:
            self.search_tree[insert_id].append({'id': img_id, 'mm_vector': mm_sign, 'sm_vector': sm_sign})

        return img_id

    def cvt_2_mosaic_gary(self, img_or_path):
        if type(img_or_path) == str:
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
        h, w = img.shape[:2]
        rate = h / 64
        new_h = int(h / rate)
        new_w = int(w / rate)
        img = cv2.resize(img, (new_w, new_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
def path_to_json(img_file_path, save_name):
    all_img_path = list_all_files(img_file_path)
    all_img_path = all_img_path[::5]
    data = []
    for img_path in all_img_path:
        base_name, parent_dir, grandparent_dir = get_parent_and_grandparent_dir(img_path)
        imid = parent_dir+ '/' + base_name
        data.append({'id': imid, 'img': img_path})
    write_json_file(save_name + '.json', data)

if __name__ == '__main__':
    import time
    save_name = 'mmsm_test01'
    #process_img_2_pkl()
    mm = MM_SSIM_map_match()
    #path_to_json('E:/Genshin_Impact/01', 'test01')
    #mm.process_img_2_pkl("test01.json", save_name + '.pkl')
    #mm.load_pkl(save_name + '.pkl')
    #mm.greedy_furthest_points(30) #001
    #mm.process_fa_node_map() #002
    #mm.save_fa_map(save_name + '_fa_map.pkl')
    #mm.load_fa_map(save_name + '_fa_map.pkl')
    #print(mm.fa_node)
    #mm.generate_search_tree() #003
    #mm.save_search_tree(save_name + '_search_tree.pkl')
    mm.load_search_tree(save_name + '_search_tree.pkl')

    start_time = time.time()
    print(mm.search_img(r'E:/Genshin_Impact/01\026340.jpg'))
    end_time = time.time()
    print(f"Search time: {end_time - start_time} seconds")
    print(mm.search_img(r'E:\Genshin_impact_circle\03\122040.jpg'))
    #print(mm.min_distance)
     
    print(mm.insert_new(r'E:\Genshin_impact_circle\03\112680.jpg'))
    print(mm.search_img(r'E:\Genshin_impact_circle\03\112680.jpg'))
    print(mm.insert_new(r'E:\Genshin_impact_circle\03\122040.jpg'))
    print(mm.search_img(r'E:\Genshin_impact_circle\03\122040.jpg'))