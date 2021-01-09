import multiprocessing
import time

def sleep(name, sleep_time, total_rsc):
    for i in range(sleep_time):
        print('%s has sleep %d / %d' % (name, i, sleep_time))
        time.sleep(1)
    total_rsc.value += 1

def show_num(name, num_list, total_rsc):
    for num in num_list:
        print('%s : %d' % (name, num))
        time.sleep(1)
    total_rsc.value += 1


if __name__ == "__main__":
    total_rsc = multiprocessing.Value("d", 2)

    tasks = [
        dict(fun=sleep, args=(0, 5, total_rsc)),
        dict(fun=sleep, args=(1, 20, total_rsc)),
        dict(fun=show_num, args=(2, list(range(5)), total_rsc)),
        dict(fun=show_num, args=(3, list(range(10)), total_rsc))
    ]
    task_id = 0
    p_list = []
    while(True):
        if task_id == len(tasks):
            break
        task = tasks[task_id]
        if total_rsc.value == 0:
            time.sleep(1)
            continue
        total_rsc.value -= 1

        p = multiprocessing.Process(target=task['fun'], args=task['args'])
        p_list.append(p)
        p_list[-1].start()

        task_id += 1


