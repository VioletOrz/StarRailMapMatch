# StarRailMapMatch
Match the map of the game "Honkai: StarRail" based on image whitout learning

# 崩坏:星穹铁道 小地图匹配
项目基于图像匹配小地图信息

# 使用说明：
1. 录制游戏的视频，录制过程中保持移动且频繁移动视角，保证采样到丰富的小地图图像信息（地图设置为固定方向
2. 将录制的视频保存为视频格式，使用tools/exct_video.py 脚本将视频转为图片，正常跑步速度下每秒采样5张图即可实现较为准确的匹配
3. 使用pipeline.py 脚本从图像中截取出合适的小地图部分用于匹配（pipeline代码中给出的预设参数适用于 2k 分辨率的图片，其中左上角起点为 60,65 外侧框边长为250，内侧圆直径为220）
4. 使用starailmapmatch.py，先将图像处理成pkl数据包，再进行匹配
    1. 使用process_img_2_pkl将图像处理为pkl数据包，points参数为1时精度最高（top1精度95%，top3精度为100%），points参数为图像数量的1/2次方时匹配效率最高（top1精度为84%）
    2. 生成数据包后，使用load_search_tree加载serach_tree, 其他两个数据包分别存储头节点和列表，仅记录数据，匹配时不会用到，后续每次匹配时不需要再重新生成数据包
    3. 使用mm.search_img/mm.search_img_topk 进行匹配 返回匹配到的图像路径和距离，距离为0-1之间，<0.6时为比较接近 <0.3时图像几乎一样
5. 关于数据收集方式
    1. 虽然可以一个视频直接跑完一张地图，但这样的话数据变化是比较均匀的，使用分块策略会造成比较大的精度损失，最好是只收集需要匹配的地标区域的小地图，跳过其他不重要的部分，可以提升精度
6. 性能
    1. onece search in 1000 images cost 0.18s (use 13900k)
    2. 13900kcpu上，在1000张图片组成的数据包中进行一次检索的最少时间需要0.18s
    3. points参数、图像总数和cpu都会影响性能
7. 致谢
    1. 这个项目的一部分匹配策略来自于 https://github.com/rhsimplex/image-match 项目，这个项目提供了非常精确且快速的图像匹配方案
8. 联系作者
    1. bilibili: https://space.bilibili.com/35596643?spm_id_from=333.1007.0.0
    2. email: violet20010528@gmail.com
    3. 测试数据很大，不会放在github上面，有需要请联系我
9. todo
    1. 优化匹配算法的性能，减少本地占用
    2. 添加新的搜索数据结构构建策略