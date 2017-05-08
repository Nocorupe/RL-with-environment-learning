# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
from random import randrange
from collections import deque
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt


if __name__ == '__main__' :
    
    if len(sys.argv) < 2 :
        print("USAGE: %s result1.npy result2.npy ...".format(sys.argv[0]))
        sys.exit()
    
    for i in range(1,len(sys.argv)) :
        r = np.load(sys.argv[i])
        if "image" in sys.argv[i] :
            plt.plot(r, "b")
        else :
            plt.plot(r, "r")

    plt.show()
       

