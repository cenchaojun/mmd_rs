from filecmp import dircmp
import os
import os.path as osp
import shutil

# 自动将旧版的代码(由compare_and_split得到)merge到新版的代码中
# 1. 将所有文件复制到新版中
# 2. 将旧版的__init__中的内容分离，放到新版中。

# dirobj = dircmp(r'..\..\..\mmd_remote_sense', r'..\..\mmdetection_org')
def mkdir(dir):
    if not os.path.exists(dir):
        # print('Mkdir %s' % dir)
        os.mkdir(dir)

def show_left_only(d):
    if len(d.left_only) > 0:
        print(r"%s  |   \nleft_only:%s" % (d.left ,str(d.left_only)))
        for sd in d.subdirs.values():
            show_left_only(sd)

def move_file(src_root, dst_root, f):
    srcpth = osp.join(src_root, f)
    dstpth = osp.join(dst_root, f)
    print((src_root, f))
    if osp.isdir(srcpth):
        if osp.exists(dstpth):
            shutil.rmtree(dstpth)
        shutil.copytree(srcpth, dstpth)
    else:
        shutil.copy(srcpth, dstpth)

def merge_init(src_root, dst_root, f):
    """
    :param src: 带有###的mmd_rs中的init
    :param dst: 新mmd的init
    :return:
    """
    src = osp.join(src_root, f)
    dst = osp.join(dst_root, f)
    with open(src, 'r') as f_src:
        lines = list(f_src.readlines())
        for i, l in enumerate(lines):
            if '########' in l:
                break
        added_part = lines[i:]
        with open(dst, 'a') as f_dst:
            for i in range(5):
                f_dst.write('   \n')
            for a in added_part:
                f_dst.write(a)

def move_left_only(left_root, right_root, dst_root, ignore_folders=None):
    """

    :param left_root: 修改后的mmd
    :param right_root: 新的mmd
    :param dst_root: 目标文件夹
    :return:
    将修改后的mmd与原始mmd进行比较，将修改的痕迹复制到目标文件夹中
    init文件保留
    """
    dst_root = right_root
    d = dircmp(left_root, right_root)
    def move_left_only_(d, dst_root):
        if len(d.left_only) == 0 and len(d.subdirs) == 0:
            return
        # if len(d.left_only) > 0 or len(d.subdirs) > 0:
        ## make dir
        if len(d.left) != len(left_root):
            dst_folder = osp.join(dst_root, d.left[len(left_root) + 1:])
        else:
            dst_folder = dst_root
        mkdir(dst_folder)



        ## move file
        if len(d.left_only) > 0:
            print("%s  |   \nleft_only:%s" % (d.left, str(d.left_only)))
            if '__init__.py' in os.listdir(d.left):
                merge_init(d.left, dst_folder, '__init__.py')
            for f in d.left_only:
                move_file(d.left, dst_folder, f)
        for sd in d.subdirs.values():
            move_left_only_(sd, dst_root)

    move_left_only_(d, dst_root)

# show_left_only(dirobj)
os.chdir('../../../')
print(os.listdir('./'))
org_folder = r'.\mmd_rs\mmdetection_org'
ignore_folders = []
modified_name = r'mmd_rs'
modified_folder = r'.\%s' % modified_name
dst_folder = r'.\compare_%s' % modified_name
mkdir(dst_folder)
move_left_only('.\compare_mmd_rs', '.\mmdetection',
               dst_folder)