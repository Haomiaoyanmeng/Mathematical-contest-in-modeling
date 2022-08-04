import pyautogui as pg
import time

print('Press Ctrl-C to quit.')
try:
    while True:
        x, y = pg.position()
        positionStr = 'X: ' + str(x) + 'Y: ' + str(y) + '\n'
        print(positionStr, end='')
        time.sleep(0.3)
except KeyboardInterrupt:
    print('111', '\n')
#
# a = 0.01
# b = 0.3
# c = 0.8
# d = 24
#
#
# pg.click(903, 1051)
#
# pg.click(903, 989)
#
#   # 开题目
# pg.click(1489, 407)
# time.sleep(0.4)
#
# pg.click(1413, 280)
# pg.write('1')
#
#
# pg.click(1279, 453)
# pg.write('hh')
# pg.hotkey(' ')
# time.sleep(0.2)
#
# # pg.moveTo(1827, 211, 0.01)
# # time.sleep(c)
# # pg.dragTo(1827, 327 + d, b, button='left')
# pg.scroll(-450)
# time.sleep(0.1)
# pg.scroll(2)
#
# pg.click(1102, 149)
# pg.click(1102, 676)
#
# # pg.moveTo(1827, 327, 0.01)
# # time.sleep(c)
# # pg.dragTo(1827, 523 + d, b, button='left')
# pg.scroll(-650)
#
# pg.scroll(2)
#
#
# pg.click(1102, 479)
#
# pg.click(1102, 620)
#
# pg.click(1102, 678)
#
# pg.click(1102, 738)
#
# pg.click(1102, 797)
#
# # pg.moveTo(1827, 523, a)
# # time.sleep(c)
# # pg.dragTo(1827, 767 + d, b, button='left')
# pg.scroll(-950)
#
# pg.scroll(2)
#
# pg.click(1102, 147)
#
# pg.click(1102, 203)
#
# pg.click(1102, 267)
#
# pg.click(1102, 444)
#
# pg.click(1102, 513)
#
# pg.click(1102, 671)
#
# pg.click(1102, 734)
#
# pg.click(1102, 797)
#
# pg.click(1102, 863)
#
# pg.click(1102, 916)
#
# # pg.moveTo(1827, 767, a)
# # time.sleep(c)
# # pg.dragTo(1827, 886 + d, b, button='left')
# pg.scroll(-700)
#
# pg.scroll(2)
#
#
# pg.click(1102, 868)
#
# # pg.dragTo(370,478, 2, button='left')
#
