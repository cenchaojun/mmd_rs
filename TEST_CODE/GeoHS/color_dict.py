rgb_colors = dict(
    White=(255, 255, 255),
    # 红色系
    DeepPink=(255, 20, 147),
    Violet=(238,130,238),
    Brown=(205,51,51),
    orange=(255,69,0),
    # 蓝色系
    BlueViolet=(138,43,226),
    Purple=(160,32,240),
    DeepSkyBlue=(0,191,255),
    # 绿色系
    Green3=(0,205,0),
    OliveDrab1=(192,255,62),
    LightGreen=(144,238,144),
    # 黄色系
    Gold=(255,215,0),
Chocolate=(210,105,30)
)

bgr_colors = {k: (v[2],v[1],v[0]) for k, v in rgb_colors.items()}