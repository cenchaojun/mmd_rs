import os
def mkdir(dir):
    if not os.path.exists(dir):
        print('Mkdir %s' % dir)
        os.mkdir(dir)
rc_root = '../mmdetection'
dst_root = './tools_test/mmd_file_list'
mkdir(dst_root)

def search_folder(folder, rc_root, dst_root,exclude_exts, include_exts,
                  generate_file_list=True):
    print(folder, rc_root, dst_root)
    files = []
    folder_name = os.path.split(folder)[1]
    if len(folder) != len(rc_root):
        dst_folder = dst_root + '/' + folder[len(rc_root) + 1:]
    else:
        dst_folder = dst_root
    mkdir(dst_folder)
    if generate_file_list:
        with open(dst_folder + '/%s_file_list.txt' % folder_name, 'wt+') as file_list:
            for f in os.listdir(folder):
                print(f)
                ext = os.path.splitext(f)[1]
                # if ext not in exclude_exts:
                # if ext in include_exts:
                file_list.write(f + '\n')
                f = folder + '/' + f

                if os.path.isdir(f):
                    search_folder(f, rc_root, dst_root, exclude_exts, include_exts,
                                  generate_file_list=generate_file_list)

    else:
       for f in os.listdir(folder):
            print(f)
            f = folder + '/' + f
            if os.path.isdir(f):
                search_folder(f, rc_root, dst_root, exclude_exts, include_exts,
                              generate_file_list=generate_file_list)

search_folder(rc_root, rc_root, dst_root,
              exclude_exts=[''],
              include_exts=['.py', '', ],
              generate_file_list=False)