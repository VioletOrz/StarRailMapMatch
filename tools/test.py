import time
from moduls.Violet_base import *
from moduls.MM_SSIM_search_tree import MM_SSIM_map_match, path_to_json

if __name__ == '__main__':
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