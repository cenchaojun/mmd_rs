import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

import time
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import os
from io import StringIO
import PIL
import cv2


def pil2ndarray(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    return img

def aa():
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(
        plt.Rectangle(
            (1, 2),  # (x,y)矩形左下角
            3,  # width长
            4,  # height宽
            color='maroon',
            alpha=0.5
        )
    )
    plt.xlim(-1, 6)
    plt.ylim(1, 7)
    plt.show()
    fig1.savefig('rect1.png', dpi=90, bbox_inches='tight')

if __name__ == "__main__":
    fig = plt.Figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.plot(range(10), [i**2 for i in range(10)])
    data = pil2ndarray(fig)
    #
    # plt.plot(range(10), [i**2 for i in range(10)])
    # data = pil2ndarray()
    cv2.imshow('image', data)
    cv2.waitKey(100000)
