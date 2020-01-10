#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : gao.py
# Author: WangYu
# Date  : 2019/12/31

import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
f = cv2.imread('201928014628024.jpg', cv2.IMREAD_COLOR)
f = cv2.filter2D(f,-1,kernel2)
plt.imshow(f)
