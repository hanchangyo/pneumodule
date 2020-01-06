# -*- coding: utf-8 -*-

# Python 3 compatibility
from __future__ import print_function
try:
    input = raw_input
except NameError:
    pass

import sys

import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.ptime import time

# from matplotlib import cm

import cv2

import copy

import sensel_control as sc

import pneumodule
import itertools

from oscpy.client import OSCClient

rows = 185
cols = 105

# interpolation ratio
interp = 3

# create Qt Application window
app = QtGui.QApplication([])
app.quitOnLastWindowClosed()

# Define a top-level widget to hold everything
win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)

# Create window with GraphicsView widget
grview = pg.GraphicsLayoutWidget()
grview.show()  # show widget alone in its own window
win.show()
win.setWindowTitle('Sensel test')
win.resize(rows * 7, cols * 7)


# set views
view1 = grview.addViewBox(row=1, col=1)  # raw frame
view2 = grview.addViewBox(row=1, col=2)  # annotations
# view3 = grview.addViewBox(row=2, col=2)  # threshold
# view4 = grview.addViewBox(row=2, col=1)  # bicubic interpolated

# lock the aspect ratio so pixels are always square
view1.setAspectLocked(True)
view2.setAspectLocked(True)
# view3.setAspectLocked(True)
# view4.setAspectLocked(True)

# Create image item
img1 = pg.ImageItem(border='y')
img2 = pg.ImageItem(border='r')
# img3 = pg.ImageItem(border='g')
# img4 = pg.ImageItem(border='b')

# add image item
view1.addItem(img1)
view2.addItem(img2)
# view3.addItem(img3)
# view4.addItem(img4)

# Set initial view bounds
view1.setRange(QtCore.QRectF(0, 0, rows * interp, cols * interp))
view2.setRange(QtCore.QRectF(0, 0, 45 * 3, 45 * 3))
# view3.setRange(QtCore.QRectF(0, 0, rows, cols))
# view4.setRange(QtCore.QRectF(0, 0, rows * interp, cols * interp))
layout.addWidget(grview, 0, 0)

# continueBtn = QtGui.QPushButton("Continue")
# layout.addWidget(continueBtn, 3, 0)

# Sensel initialization
handle, info = sc.open_sensel()

# Initalize frame
frame = sc.init_frame(handle, detail=0, baseline=0)

# update interval
interval = 0  # miliseconds

lastTime = time()
fps = None

BlobTracker = pneumodule.TrackBlobs(interp)

marker_size = 45  # 60 mm main module
magnet_dist = 34  # 60 mm module
magnet_offset = 2  # for 60 mm module

MarkerTracker = pneumodule.TrackMarkers(marker_size, magnet_dist, magnet_offset)

address = '127.0.0.1'
# address = '192.168.0.3'
port = 12000

osc = OSCClient(address, port)
osc_sub = OSCClient(address, 12001)


def update():
    global lastTime, fps, info, handle, frame, f_image, do_save
    try:
        f_image = sc.scan_frames(handle, frame, info)
    except UnboundLocalError:
        sc.close_sensel(handle, frame)
        # Sensel initialization
        handle, info = sc.open_sensel()
        # Initalize frame
        frame = sc.init_frame(handle, detail=0, baseline=0)
        f_image = sc.scan_frames(handle, frame, info)

    # print(np.max(f_image))

    # prepare image to show
    f_image_show = copy.deepcopy(f_image)
    if np.max(f_image_show) > 0:
        f_image_show = f_image_show / np.max(f_image_show) * 255
    f_image_show = cv2.cvtColor(f_image_show.astype(np.uint8),
                                cv2.COLOR_GRAY2RGB
                                )

    # # find blobs from the image
    # blobs, contours, hierarchy, areas, cx, cy, forces, f_image_thre, f_image_interp = pneumodule.detectBlobs(f_image, areaThreshold=1000, interp=interp)
    # print(contours)

    # interpolate image
    f_image_interp = cv2.resize(f_image, (f_image.shape[1] * interp, f_image.shape[0] * interp), interpolation=cv2.INTER_LANCZOS4)

    if np.max(f_image_interp) > 0:
        f_image_interp = f_image_interp / np.max(f_image_interp) * 255
    f_image_interp = cv2.cvtColor(f_image_interp.astype(np.uint8),
                                  cv2.COLOR_GRAY2RGB
                                  )

    # update blob information
    blobs = BlobTracker.update(f_image)

    # update marker information
    MarkerTracker.update(blobs, f_image)
    if len(MarkerTracker.markers) > 0:
        marker_num = 0
        for mkr in MarkerTracker.markers:
            # print('life time:', mkr.lifetime)
            mkr_center = (np.int(mkr.pos_x * interp), np.int(mkr.pos_y * interp))
            # cv2.circle(
            #     f_image_interp,
            #     mkr_center,
            #     30,
            #     (255, 255, 255)
            # )
            # cv2.line(f_image_interp, mkr_center, (int(mkr_center[0] + mkr.rot[0] * 40), int(mkr_center[1] + mkr.rot[1] * 40)), (255, 255, 255), 3)
            cv2.arrowedLine(f_image_interp,
                            (int(mkr_center[0] + mkr.rot[0] * mkr.marker_size / 2 * interp), int(mkr_center[1] + mkr.rot[1] * mkr.marker_size / 2 * interp)),
                            (int(mkr_center[0] + mkr.rot[0] * mkr.marker_size / 2 * interp * 1.5), int(mkr_center[1] + mkr.rot[1] * mkr.marker_size / 2 * interp * 1.5)),
                            (255, 255, 255), thickness=5, tipLength=0.5)
            # for b in mkr.blobs:
            #     if b is not None:
            #         cv2.circle(
            #             f_image_interp,
            #             (np.int(b.sub_cx * interp), np.int(b.sub_cy * interp)),
            #             7,
            #             (255, 255, 0)
            #         )
            for b in mkr.desired_blob_pos:
                if b is not None:
                    cv2.circle(
                        f_image_interp,
                        (np.int(b[0] * interp), np.int(b[1] * interp)),
                        10,
                        (255, 0, 0)
                    )

            # draw marker border
            for border_pair in itertools.combinations(mkr.marker_border, 2):
                c1 = (int(border_pair[0][0] * interp), int(border_pair[0][1]) * interp)
                c2 = (int(border_pair[1][0] * interp), int(border_pair[1][1]) * interp)
                # dist = pneumodule.distance(c1, c2)
                # print(dist)
                if pneumodule.distance(c1, c2) < mkr.marker_size * interp * 1.2:
                    cv2.line(f_image_interp, c1, c2, (255, 255, 0), 1)

            # show marker region
            crop_img_show = copy.deepcopy(mkr.crop_img)
            if np.max(crop_img_show) > 0:
                crop_img_show = crop_img_show / np.max(crop_img_show) * 255
            crop_img_show = cv2.cvtColor(crop_img_show.astype(np.uint8),
                                         cv2.COLOR_GRAY2RGB
                                         )
            # display force
            # for sub_r in range(1):
            #     for i in range(3):
            #         for j in range(3):
            #             if sub_r == 0:
            #                 upperleft = [5 * 3 * interp, 5 * 3 * interp]
            #             else:
            #                 upperleft = [5 * 3 * interp, 5 * 3 * interp]
            #             font = cv2.FONT_HERSHEY_SIMPLEX
            #             cv2.putText(
            #                 crop_img_show,
            #                 '%0.0f' % mkr.pressure_values[sub_r][i][j],
            #                 (int(upperleft[0] + j * interp * 5), int(upperleft[0] + i * interp * 5)),
            #                 font,
            #                 0.3,  # font size
            #                 (0, 255, 0),
            #                 1,
            #                 cv2.LINE_AA
            #             )

            # send osc messages
            # center_x = []
            # center_y = []
            # for i in centers:
            #     center_x.append(i[0])
            #     center_y.append(i[1])
            pos_x = b'/marker/%s/pos_x' % bytes(str.encode(str(marker_num)))
            pos_y = b'/marker/%s/pos_y' % bytes(str.encode(str(marker_num)))
            angle = b'/marker/%s/angle' % bytes(str.encode(str(marker_num)))
            osc.send_message(pos_x, [mkr.pos_x])
            osc_sub.send_message(pos_x, [mkr.pos_x])
            osc.send_message(pos_y, [mkr.pos_y])
            osc_sub.send_message(pos_y, [mkr.pos_y])
            osc.send_message(angle, [mkr.rot_rad])
            osc_sub.send_message(angle, [mkr.rot_rad])
            for sub_r in range(5):
                id_list = [0] * 4
                for i in range(3):
                    for j in range(3):
                        msg = b'/marker/%s/force/%s/%s' % (bytes(str.encode(str(marker_num))), bytes(str.encode(str(sub_r))), bytes(str.encode(str(i * 3 + j))))
                        osc.send_message(msg, [mkr.pressure_values[sub_r][i][j]])
                        osc_sub.send_message(msg, [mkr.pressure_values[sub_r][i][j]])

                        # determine id
                        if i == 0 and j == 0:
                            if mkr.pressure_values[sub_r][i][j] > 100:
                                id_list[0] = 1
                        if i == 0 and j == 2:
                            if mkr.pressure_values[sub_r][i][j] > 100:
                                id_list[1] = 1
                        if i == 2 and j == 0:
                            if mkr.pressure_values[sub_r][i][j] > 100:
                                id_list[2] = 1
                        if i == 2 and j == 2:
                            if mkr.pressure_values[sub_r][i][j] > 100:
                                id_list[3] = 1
                # print(id_list)
                msg = b'/marker/%s/id/%s' % (bytes(str.encode(str(marker_num))), bytes(str.encode(str(sub_r))))
                if id_list == [1, 0, 0, 0]:
                    osc.send_message(msg, [1])
                    osc_sub.send_message(msg, [1])
                elif id_list == [0, 1, 0, 0]:
                    osc.send_message(msg, [2])
                    osc_sub.send_message(msg, [2])
                elif id_list == [0, 1, 0, 0]:
                    osc.send_message(msg, [2])
                    osc_sub.send_message(msg, [2])
                elif id_list == [0, 0, 1, 0]:
                    osc.send_message(msg, [3])
                    osc_sub.send_message(msg, [3])
                elif id_list == [0, 1, 1, 0]:
                    osc.send_message(msg, [4])
                    osc_sub.send_message(msg, [4])
                elif id_list == [1, 0, 1, 0]:
                    osc.send_message(msg, [5])
                    osc_sub.send_message(msg, [5])
                elif id_list == [1, 1, 0, 0]:
                    osc.send_message(msg, [6])
                    osc_sub.send_message(msg, [6])
                elif id_list == [0, 0, 0, 1]:
                    osc.send_message(msg, [7])
                    osc_sub.send_message(msg, [7])
                if id_list == [1, 0, 0, 1]:
                    osc.send_message(msg, [8])
                    osc_sub.send_message(msg, [8])
                elif id_list == [0, 1, 0, 1]:
                    osc.send_message(msg, [9])
                    osc_sub.send_message(msg, [9])
                elif id_list == [0, 1, 0, 1]:
                    osc.send_message(msg, [10])
                    osc_sub.send_message(msg, [10])
                elif id_list == [0, 0, 1, 1]:
                    osc.send_message(msg, [11])
                    osc_sub.send_message(msg, [11])
                elif id_list == [0, 1, 1, 1]:
                    osc.send_message(msg, [12])
                    osc_sub.send_message(msg, [12])
                elif id_list == [1, 0, 1, 1]:
                    osc.send_message(msg, [13])
                    osc_sub.send_message(msg, [13])
                elif id_list == [1, 1, 0, 1]:
                    osc.send_message(msg, [14])
                    osc_sub.send_message(msg, [14])
                elif id_list == [0, 0, 0, 0]:
                    osc.send_message(msg, [0])
                    osc_sub.send_message(msg, [0])

            marker_num += 1

            img2.setImage(np.rot90(crop_img_show, 3), autoLevels=True, levels=(0, 80))
            # print(mkr.pressure_values[0].astype(np.int))
            # img2.setImage(np.rot90(mkr.sub_region, 3), autoLevels=True, levels=(0, 80))
            # print(np.sum(mkr.sub_region))
            # print(mkr.sub_region[0:2, 0:2])
            # print(mkr.sub_region[0 * 5 * 3 - 1:(0 + 1) * 5 * 3, 0 * 5 * 3 - 1:(0 + 1) * 5 * 3])


                # c1 = (int(mkr.blobs[1].sub_cx * interp), int(mkr.blobs[1].sub_cy * interp))
                # c2 = (int(mkr.blobs[2].sub_cx * interp), int(mkr.blobs[2].sub_cy * interp))
                # cv2.line(f_image_interp, c1, c2, (255, 255, 0), 1)

    # display force
    if len(blobs) > 0:
        for b in blobs:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                f_image_interp,
                '%0.0f' % b.force,
                (int(b.cx * interp) + 5, int(b.cy * interp) + 10),
                font,
                0.5,  # font size
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    # # display appeared time
    # if len(blobs) > 0:
    #     for b in blobs:
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(
    #             f_image_show,
    #             '%5.2f' % b.t_appeared,
    #             (int(b.cx), int(b.cy) + 10),
    #             font,
    #             0.3,  # font size
    #             (0, 255, 255),
    #             1,
    #             cv2.LINE_AA
    #         )

    # display ID
    # if len(blobs) > 0:
    #     for b in blobs:
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(
    #             f_image_show,
    #             'ID: %d' % b.ID,
    #             (int(b.cx) + 5, int(b.cy) + 0),
    #             font,
    #             0.3,  # font size
    #             (255, 255, 255),
    #             1,
    #             cv2.LINE_AA
    #         )

    # show peaks
    if len(blobs) > 0:
        for b in blobs:
            cv2.circle(
                f_image_interp,
                (np.int(b.sub_cx * interp), np.int(b.sub_cy * interp)),
                3,
                (255, 0, 0)
            )

    # # display coords
    # if len(blobs) > 0:
    #     for b in blobs:
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(
    #             f_image_interp,
    #             '(%0.2f, %0.2f)' % (b.sub_cx, b.sub_cy),
    #             (int(b.cx * interp) + 5, int(b.cy * interp) + 0),
    #             font,
    #             0.5,  # font size
    #             (255, 255, 255),
    #             1,
    #             cv2.LINE_AA
    #         )

    # img1.setImage(np.rot90(f_image, 3), autoLevels=True, levels=(0, 50))
    # img1.setImage(np.rot90(f_image_peaks, 3), autoLevels=True, levels=(0, 50))
    # adjust sensitivity

    # img1.setImage(np.rot90(f_image, 3), autoLevels=False, levels=(0, 255))
    # img2.setImage(np.rot90(f_image_show, 3), autoLevels=True, levels=(0, 80))
    # img3.setImage(np.rot90(f_image_thre, 3), autoLevels=True, levels=(0, 80))
    img1.setImage(np.rot90(f_image_interp, 3), autoLevels=True, levels=(0, 80))
    # img1.setImage(np.rot90(f_image_interp, 3), autoLevels=True, levels=(0, 80))

    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0 / dt
    else:
        s = np.clip(dt * 3., 0, 1)
        fps = fps * (1 - s) + (1.0 / dt) * s

    print('%0.2f fps' % fps)
    QtGui.QApplication.processEvents()


# update using timer
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(interval)

if __name__ == "__main__":
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        print('Closed the window')
        sc.close_sensel(handle, frame)
