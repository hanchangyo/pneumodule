# -*- coding: utf-8 -*-

# Python 3 compatibility
from __future__ import print_function
try:
    input = raw_input
except NameError:
    pass

import sys
sys.path.append('sensel-lib-python')
import sensel
import numpy as np


def open_sensel():
    handle = None
    error, device_list = sensel.getDeviceList()
    if device_list.num_devices != 0:
        error, handle = sensel.openDeviceByID(device_list.devices[0].idx)
    error, info = sensel.getSensorInfo(handle)
    print(info)

    return handle, info


def init_frame(handle, setrate=125, detail=0, baseline=1, devbuffer=10):

    # Frame rate
    # setrate = 2000
    error = sensel.setMaxFrameRate(handle, setrate)

    # Scan detail
    # 0 for high, 1 for medium, 2 for low
    # detail = 0
    error = sensel.setScanDetail(handle, detail)

    error, rate = sensel.getMaxFrameRate(handle)
    print('Rate: ' + str(rate))
    error, val = sensel.getScanDetail(handle)
    print('Scan detail: ' + str(val))

    # Misc
    error = sensel.setDynamicBaseline(handle, baseline)
    error, val = sensel.getDynamicBaseline(handle)
    print('Dynamic baseline: ' + str(val))

    error = sensel.setEnableBlobMerge(handle, 0)
    error, val = sensel.getEnableBlobMerge(handle)
    print('Blob merge: ' + str(val))

    # Frame content
    # The Content Bitmask is used to enable or disable reporting of the five (5) parameters in Frame Data:
    # Contacts, Force Array, Labels Array, Accelerometer Data, and Lost Frame Count.
    # Set the byte value according to the parameters needed. For example,
    # FRAME_CONTENT_PRESSURE_MASK = 0x01
    # FRAME_CONTENT_LABELS_MASK   = 0x02
    # FRAME_CONTENT_CONTACTS_MASK = 0x04
    # FRAME_CONTENT_ACCEL_MASK    = 0x08

    # mask = sensel.FRAME_CONTENT_PRESSURE_MASK + sensel.FRAME_CONTENT_ACCEL_MASK
    # mask = sensel.FRAME_CONTENT_PRESSURE_MASK + sensel.FRAME_CONTENT_CONTACTS_MASK
    mask = sensel.FRAME_CONTENT_PRESSURE_MASK
    # mask = sensel.FRAME_CONTENT_CONTACTS_MASK
    error = sensel.setFrameContent(handle, mask)

    # set buffer control
    error = sensel.setBufferControl(handle, devbuffer)
    error, val = sensel.getBufferControl(handle)
    print('Device buffer: ' + str(val))

    error, frame = sensel.allocateFrameData(handle)
    error = sensel.startScanning(handle)
    return frame


def scan_frames(handle, frame, info):
    error = sensel.readSensor(handle)
    error, num_frames = sensel.getNumAvailableFrames(handle)
    # print('Available num frames:', num_frames)
    f_image = np.zeros((info.num_rows, info.num_cols))
    for i in range(num_frames):
        error = sensel.getFrame(handle, frame)
        # print('Content bit mask: ', frame.content_bit_mask)
        f_image = print_frame(frame, info)
    return f_image


def print_frame(frame, info):
    f_array = np.reshape(
        np.asarray(frame.force_array[:info.num_rows * info.num_cols]),
        (info.num_rows, info.num_cols)
    )

    return f_array


def close_sensel(handle, frame):
    error = sensel.freeFrameData(handle, frame)
    error = sensel.stopScanning(handle)
    error = sensel.close(handle)
