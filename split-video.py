# This is a sample Python script.
import cv2
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):

    vidcap = cv2.VideoCapture('video/test.mov')
    success, image = vidcap.read()
    count = 0
    imgcount = 1
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        cv2.imwrite("video/test2/clicked-%d.png" % imgcount, image)  # save frame as PNG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 3.5
        imgcount += 1


print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/