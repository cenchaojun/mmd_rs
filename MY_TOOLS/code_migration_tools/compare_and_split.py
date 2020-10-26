from filecmp import dircmp
import os
import os.path as osp
import shutil


dirobj = dircmp(r'..\mmd_remote_sense', r'..\mmdetection')
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
    # if os.system('copy %s %s' % (srcpth, dstpth)) == 0:
    #     print('copy %s %s' % (srcpth, dstpth))
    #     print('COPY DONE')
    #     pass
    # else:
    #     print('copy %s %s' % (srcpth, dstpth))
    #     print('COPY FAILED')

def move_left_only(left_root, right_root, dst_root):
    """

    :param left_root: 修改后的mmd
    :param right_root: 原始mmd
    :param dst_root: 目标文件夹
    :return:
    将修改后的mmd与原始mmd进行比较，将修改的痕迹复制到目标文件夹中
    init文件保留
    """
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
                move_file(d.left, dst_folder, '__init__.py')
            for f in d.left_only:
                move_file(d.left, dst_folder, f)
        for sd in d.subdirs.values():
            move_left_only_(sd, dst_root)

    move_left_only_(d, dst_root)

# show_left_only(dirobj)
org_folder = r'..\mmdetection'
modified_name = r'mmd_rs_ad'
modified_folder = r'..\%s' % modified_name
dst_folder = r'.\tools_test\compare_%s' % modified_name
mkdir(dst_folder)
move_left_only(modified_folder, org_folder,
               dst_folder)