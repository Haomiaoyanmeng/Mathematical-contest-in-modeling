#!C:\Users\86138\AppData\Local\Programs\Python\Python38-32
import pyautogui as pg
import time
import numpy as np

# x1 = 0
# y1 = 0
#
# i = 0
# flag = 0
# posi_sto = np.array([[-1, 0, 0]])
# try:
#     while True:
#         x, y = pg.position()
#         positionStr = 'X: ' + str(x) + 'Y: ' + str(y)
#         # print(positionStr, end='')
#         time.sleep(0.3)
#         if x1 == x and y1 == y:
#             i = i + 1
#         else:
#             i = 0
#             flag = 0
#         print(positionStr, '  i=', i, '\n')
#         x1 = x
#         y1 = y
#
#         if i >= 5 and flag == 0:
#             flag = 1
#             posi_sto = np.row_stack((posi_sto, [x, y, int(i / 5)]))
#
# except KeyboardInterrupt:
#     print('111', '\n')
#
# print(posi_sto[1:, :], end=',')
# # print(",".join(str(i) for i in posi_sto[1:, :]))

click_sto = [[1858,  910,    2],
 [ 1000, 62,    2],
 [ 1000,  129,    3],
 [ 957,  929,    1],
[1339, 686, 1],
 [ 248,  214,    1],
 [ 114, 1018,    1],
 [26, 855, 4],
 [  26,  855,    1],
 [ 951,  942,    1],
 [1888,   14,    1]]

for i in range(len(click_sto)):
    if click_sto[i][2] == 2:
        pg.doubleClick(click_sto[i][0], click_sto[i][1])
        if i == 0:
            time.sleep(2)
        if i == 1:
            pg.click(click_sto[i][0], click_sto[i][1])
            # time.sleep(3)
    elif click_sto[i][2] == 1:
        pg.click(click_sto[i][0], click_sto[i][1])
    elif click_sto[i][2] == 3:
        pg.write('https://student.hitsz.edu.cn/xg_mobile/loginChange')
        time.sleep(0.5)
        # pg.keyDown('enter')
        # time.sleep(0.5)
        # pg.keyUp('enter')
        pg.click(click_sto[i][0], click_sto[i][1])
    elif click_sto[i][2] == 4:
        for j in range(10):
            pg.scroll(-1000)
            time.sleep(0.3)
    time.sleep(3)

input()
