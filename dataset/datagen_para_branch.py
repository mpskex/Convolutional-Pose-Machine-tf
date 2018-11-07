#coding: utf-8
""" Fangrui Liu
    Paralleled Datagen wrapper

    Using one line to make your training process paralleled!

    iLab at Department of Computer Science & Technology
    Beijing University of Technology

    24/10/2018
"""
import time
from multiprocessing import Pool, Process, Manager, Semaphore

class DataGenParaWrapper(object):
    """ Paralleled Data Generator Parallel Wrapper
        This would provide you a asynchronized quene with locks
    """
    def __init__(self, datagen, buff_size=4):
        self.datagen = datagen
        self.buff_size = buff_size
        self.manager = Manager()
        self.batch_list = {}
        self.gen_num = 0
        #   wrap all attribute to this wrapper
        #   except generator
        for n in dir(datagen):
            if '__' not in n and n !='generator':
                setattr(self, n, getattr(datagen, n))
    
    def task(self, gen, number, target_remain, buff_count):
        """ worker keep the list upto <buff_size>
        """
        #   prevent the generator get the empty list
        print '[*]\tWorker is going up'
        while True:
            target_remain.acquire(block=True)
            self.batch_list[number].append(next(gen))
            buff_count.release()

    def generator(self, *args, **kwargs):
        """ This function warp generator to ParaWrapper's generator
            which is capable of multi-processing
            Once the generator function was settled, we can send worker with the task then
            work with full-load until meet the buff_size limit

            The worker's job is to feed the list and keep it contains more than <buff_size> batches
        """
        #   Initialization semaphores and numbering
        buff_count = Semaphore(value=0)
        target_remain = Semaphore(value=self.buff_size)
        number = str(self.gen_num)
        self.gen_num += 1

        #   Initializing list
        self.batch_list[number] = self.manager.list()

        #   Assign work and send worker
        gen = self.datagen.generator(*args, **kwargs)
        worker = Process(target=self.task, args=(gen,number,target_remain, buff_count))
        worker.start()

        while True:
            buff_count.acquire(block=True)
            ret = self.batch_list[number].pop()
            target_remain.release()
            yield ret



class DataGenDebug(object):
    def __init__(self):
        self.k = 1
    def generator(self, a, b=1):
        while True:
            yield a+b

if __name__ == '__main__':
    datagen = DataGenDebug()
    wrapper = DataGenParaWrapper(datagen, buff_size=20)
    gen_b = wrapper.generator(1, b=2)
    gen_a = wrapper.generator(4, b=7)
    print wrapper.k
    for n in range(1000):
        print next(gen_a)
        print next(gen_b)