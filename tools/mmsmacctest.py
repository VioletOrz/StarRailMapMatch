
from moduls.SSIM import calculate_ssim
from moduls.Violet_base import list_all_files, write_pkl_file, read_pkl_file, get_parent_and_grandparent_dir, read_json_file,write_json_file
from moduls.MM_SSIM_search_tree import *

if __name__ == '__main__':
    import time
    st = time.time()
    save_name = 'mmsm_SR08'
    mm = MM_SSIM_map_match()

    # print(f"Total time: {et - st} seconds")

    mm.load_pkl(save_name + '.pkl')
    mm.load_fa_map(save_name + '_fa_map.pkl')
    mm.load_search_tree(save_name + '_search_tree.pkl')

    paths = list_all_files('./output_SR')\
    
    true = 0
    false = 0
    tobe = 0
    onetrue = 0
    for i, p in enumerate(paths):
        if (true + false + tobe) > 0:
            print(f"true: {true}, false: {false}, tobe: {tobe}")
            print(f"acc: {true / (true + false + tobe)}")
            print(f"precision: {onetrue / true}")
        print("#"*100)
        print(i,p)
        print("="*100)
        id = int(p.split('\\')[-1][-9:-4])
        if id % 5 == 0:
            continue

        mpl = mm.search_img_topk(p)
        print(mpl)

        print(f"topk长度为:{len(mpl)}")
        if len(mpl) == 0:
            false += 1
            continue
        
        if abs(int(mpl[0][0].split('\\')[-1][-9:-4]) - id) < 5:
            print("con 1 ture")
            true += 1
            onetrue += 1
            continue

        elif len(mpl) < 2:
            print('con 1 false')
            false += 1
            continue

        if len(mpl) >= 2:
            if len(mpl) == 2:
                id1 = int(mpl[0][0].split('\\')[-1][-9:-4])
                id2 = int(mpl[1][0].split('\\')[-1][-9:-4])
                print(f'{p} 搜索结果为:{mpl}')
                if (abs(id1 - id2) == 5) and ((id1 > id and id2 < id) or (id1 < id and id2 > id)):
                    print("con 1")
                    true += 1
                    continue
                elif (id1 > id and id2 < id) or (id1 < id and id2 > id):
                    print("con 2")
                    tobe += 1
                    continue
                else:
                    print("con 3")
                    false += 1
                    continue
            if len(mpl) > 2:
                id1 = int(mpl[0][0].split('\\')[-1][-9:-4])
                id2 = int(mpl[1][0].split('\\')[-1][-9:-4])
                tmp_list = []
                tmp_list.append(id)
                for tm in mpl[:3]:
                    tmp_list.append(int(tm[0].split('\\')[-1][-9:-4]))
                sorted_list = sorted(tmp_list)
                if id == sorted_list[0] or id == sorted_list[-1]:
                    false += 1
                    print("con 4")
                    continue
                elif (abs(id1 - id2) == 5) and ((id1 > id and id2 < id) or (id1 < id and id2 > id)):
                    print("con 5")
                    true += 1
                    continue
                else:
                    print("con 6")
                    tobe += 1
                    continue
print(f"true: {true}, false: {false}, tobe: {tobe}")
print(f"acc: {true / (true + false + tobe)}")
print(f"precision: {onetrue / true}")