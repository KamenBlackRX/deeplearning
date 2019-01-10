#!/usr/bin/env python
import cProfile

class Benchmark(object):

    def __init__(self, *args, **kargs):
        print ("Status: ")

    def profileFunc(self,func):
        cProfile.run(func)

    