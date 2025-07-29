from moduls.SSIM import calculate_ssim
from moduls.Violet_base import list_all_files, write_pkl_file, read_pkl_file, get_parent_and_grandparent_dir, read_json_file,write_json_file
import cv2
import random
from image_match.goldberg import ImageSignature
from moduls.MM_SSIM_search_tree import *

def process_img_2_pkl(images_path,points, save_name):
    mm = MM_SSIM_map_match()
    path_to_json(images_path, 'SR_08')
    mm.process_img_2_pkl("SR_08.json", save_name + '.pkl')
    
    mm.kdtree_furthest_points_static(33)

    mm.greedy_furthest_points(points) #001 #效果好但效率低 points的最佳数值为图像数量的1/2次方
    # mm.kmeans_plus_centroids(33) #效率低但效果好
    mm.process_fa_node_map() #002
    mm.save_fa_map(save_name + '_fa_map.pkl')
    
    print(mm.fa_node)
    mm.generate_search_tree() #003
    mm.save_search_tree(save_name + '_search_tree.pkl')
    et = time.time()
    print(f"Total time: {et - st} seconds")
    mm.calculate_centroid_distances()

if __name__ == '__main__':
    import time
    st = time.time()
    save_name = 'pkl/mmsm_SR06'
    images_path = "./output_SR"
    points = 33 # points的最佳数值为图像数量的1/2次方
    mm = MM_SSIM_map_match()
    # process_img_2_pkl(images_path, points, save_name)


    # mm.load_pkl(save_name + '.pkl')
    # mm.load_fa_map(save_name + '_fa_map.pkl')
    mm.load_search_tree(save_name + '_search_tree.pkl')

    start_time = time.time()
    print(mm.search_img(r'output_SR\central_circle_output_frame_03297.png'))
    end_time = time.time()
    print(f"Search time: {end_time - start_time} seconds")
    start_time = time.time()
    print(mm.search_img_topk(r'output_SR\central_circle_output_frame_03298.png', ))
    end_time = time.time()
    print(f"Search time: {end_time - start_time} seconds")