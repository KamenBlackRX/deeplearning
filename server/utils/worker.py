import multiprocessing as mp
import time
import os

class Workers(object):

    numWorkers = None
    initState = None
    StopID = None
    TaskList = None
    _Pool = None

    def __init__(self, *arg, **kargs):
        self.numWorkers = kargs['numWorkers']
        self.initState = kargs['initState']
        self.StopID = kargs['StopID']
        self.ProcessList = kargs['ProcessList']
        self._Pool = mp.Pool(processes=self.numWorkers)

    def enqueue_in_list(self, taskList=None, argsList=None):
        """
            Enqueue all process and run it after queue.
            @param processList The process list made in a dict.
            @returns The result list from each Task.
        """
        result = None
        assert taskList is not None, 'TaskList need be full'
        
        for i in taskList:
            for arg in argsList:
                result += self._Pool.apply_async(i, args=(arg,) )
        return result

    def enqueue_in_single(self, Task, **kargs):
        result = self._Pool.starmap(Task, kargs)
        return result

    