import json
from matplotlib import pyplot


def load_log(path):
    with open(path) as f:
        # read json file
        ch_log = json.load(f)

    x = [ep['epoch'] for ep in ch_log]
    y1 = [ep['validation/main/accuracy'] for ep in ch_log]
    y2 = [ep['main/accuracy'] for ep in ch_log]
    return x, y1, y2


def plot(x, y1, y2, title=None):
    lines = pyplot.plot(x, y1, label='v/m/accuracy')
    # ラインのスタイルを変更する
    # lines[0].set_color('#FF0000')  # 赤色に
    # lines[0].set_linestyle('-')
    # lines[0].set_linewidth(1)  # 線幅

    lines = pyplot.plot(x, y2, label='m/accuracy')

    pyplot.legend()  # 凡例表示
    pyplot.xlabel("epoch")
    pyplot.ylabel("accurancy")

    if title:
        pyplot.title(title)
    pyplot.grid(True)
    # 描画
    pyplot.show()

d = load_log('log')
plot(*d)