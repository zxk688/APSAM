
import numpy as np
import time
import sys
import numpy as np
np.bool = np.bool_
class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

# 一个用来存储并计算某些值的平均值和计数的实用类。它常用于训练过程中追踪损失、精度等指标的变化
class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

# 一个用于跟踪和测量时间的实用类，常用于训练和验证过程中记录时间消耗以及预估剩余时间
class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_est_finish(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out


def to_one_hot(sparse_integers, maximum_val=None, dtype=np.bool):

    if maximum_val is None:
        maximum_val = np.max(sparse_integers) + 1

    src_shape = sparse_integers.shape

    flat_src = np.reshape(sparse_integers, [-1])
    src_size = flat_src.shape[0]

    one_hot = np.zeros((maximum_val, src_size), dtype)
    one_hot[flat_src, np.arange(src_size)] = 1

    one_hot = np.reshape(one_hot, [maximum_val] + list(src_shape))

    return one_hot
    
    
from multiprocessing.pool import ThreadPool

class BatchThreader:

    def __init__(self, func, args_list, batch_size, prefetch_size=4, processes=12):
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size

        self.pool = ThreadPool(processes=processes)
        self.async_result = []

        self.func = func
        self.left_args_list = args_list
        self.n_tasks = len(args_list)

        # initial work
        self.__start_works(self.__get_n_pending_works())


    def __start_works(self, times):
        for _ in range(times):
            args = self.left_args_list.pop(0)
            self.async_result.append(
                self.pool.apply_async(self.func, args))


    def __get_n_pending_works(self):
        return min((self.prefetch_size + 1) * self.batch_size - len(self.async_result)
                   , len(self.left_args_list))



    def pop_results(self):

        n_inwork = len(self.async_result)

        n_fetch = min(n_inwork, self.batch_size)
        rtn = [self.async_result.pop(0).get()
                for _ in range(n_fetch)]

        to_fill = self.__get_n_pending_works()
        if to_fill == 0:
            self.pool.close()
        else:
            self.__start_works(to_fill)

        return rtn




def get_indices_of_pairs(radius, size):

    search_dist = []

    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    radius_floor = radius - 1

    full_indices = np.reshape(np.arange(0, size[0]*size[1], dtype=np.int64),
                                   (size[0], size[1]))

    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor],
                              [-1])

    indices_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)

    return indices_from, concat_indices_to
