import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
import itertools
import time
import cv2
from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull as ConvexHull
import copy


def findLocalPeaks(img, threshold=0.5, kernal=3):
    # apply the local maximum filter; all pixel of maximum value
    # in their neighborhood are set to 1
    local_max_g = maximum_filter(img, kernal)
    local_min_g = minimum_filter(img, kernal)

    # store local maxima
    local_max = (local_max_g == img)

    # difference between local maxima and minima
    diff = ((local_max_g - local_min_g) > threshold)
    # insert 0 where maxima do not exceed threshold
    local_max[diff == 0] = 0

    return local_max


def isDotIncluded(dot, rows=185, cols=105):
    # check if the dot is in the pad area
    if dot[0] > 0 and dot[0] < rows and dot[1] > 0 and dot[1] < cols:
        return True
    else:
        return False


def distance(pt1, pt2):
    # calculate distance between two points
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def findPeakCoord(img):
    # return peak coordinates from input peak image
    peaks = [tuple(coords) for coords in zip(*np.where(img == True))]
    # print(peaks)
    return peaks


def findSubpixelPeaks(peaks, img, n=5):
    # prepare an empty array of kernal size
    # n = 7
    subPeaks = []
    for pk in peaks:
        r = np.floor(n / 2)
        # crop region around peak coord with the kernal size
        cropped = cropImage(img, pk, radius=r, margin=0)

        # project values to x axis and y axis each
        x = np.sum(cropped, axis=0)
        y = np.sum(cropped, axis=1)

        # perform CoM peak detection
        # x_CoM7 = (3 * x[6] + 2 * x[5] + x[4] - x[2] - 2 * x[1] - 3 * x[0]) / (np.sum(x))
        # y_CoM7 = (3 * y[6] + 2 * y[5] + y[4] - y[2] - 2 * y[1] - 3 * y[0]) / (np.sum(y))
        if n == 7:
            x_CoM = (3 * x[6] + 2 * x[5] + x[4] - x[2] - 2 * x[1] - 3 * x[0]) / (np.sum(x))
            y_CoM = (3 * y[6] + 2 * y[5] + y[4] - y[2] - 2 * y[1] - 3 * y[0]) / (np.sum(y))
        elif n == 5:
            x_CoM = (2 * x[4] + 1 * x[3] - 1 * x[1] - 2 * x[0]) / (np.sum(x))
            y_CoM = (2 * y[4] + 1 * y[3] - 1 * y[1] - 2 * y[0]) / (np.sum(y))

        # print(x_CoM, y_CoM)
        subPeaks.append((pk[0] + x_CoM, pk[1] + y_CoM))

    return subPeaks


def constraint(input, const_floor, const_ceil):
    # make input to be
    # const_floor < input < const_ceil
    if input < const_floor:
        input = const_floor
    elif input > const_ceil:
        input = const_ceil
    else:
        input = input
    return input


def detectDots(img, coords, area=4):
    # detect if there are peaks in dot candidates with kernals
    # for i in range(area):
        # for j in range(area):
    x_start = coords[0] - int(area / 2)
    y_start = coords[1] - int(area / 2)
    x_end = coords[0] + int(area / 2) + 1
    y_end = coords[1] + int(area / 2) + 1
    x_start = constraint(x_start, 0, np.shape(img)[0])
    y_end = constraint(y_end, 0, np.shape(img)[1])
    x_start = constraint(x_start, 0, np.shape(img)[0])
    y_end = constraint(y_end, 0, np.shape(img)[1])
    kernal = img[x_start:x_end, y_start:y_end]
    # print(kernal)
    # print(np.sum(kernal))
    if np.sum(kernal) >= 1:
        return 1
    else:
        return 0


def cropImage(img, pos, radius, margin=4):
    # crop image region surrounded by circle

    # img size
    width = np.shape(img)[0]
    height = np.shape(img)[1]

    posX = int(pos[0])
    posY = int(pos[1])

    # crop size
    crop = int(round((radius + margin)) * 2 + 1)
    crop_half = int(round(radius) + margin)

    imgCropped = np.zeros((crop, crop))

    xMinFixed = 0
    xMaxFixed = crop
    yMinFixed = 0
    yMaxFixed = crop

    xMin = posX - crop_half
    xMax = posX + crop_half
    yMin = posY - crop_half
    yMax = posY + crop_half

    if xMin < 0:
        xMinFixed = -xMin
    if xMax >= width:
        xMaxFixed = crop + (width - xMax) - 1
    if yMin < 0:
        yMinFixed = -yMin
    if yMax >= height:
        yMaxFixed = crop + (height - yMax) - 1

    pxMin = constraint(posX - crop_half, 0, width)
    pxMax = constraint(posX + crop_half, 0, width)
    pyMin = constraint(posY - crop_half, 0, height)
    pyMax = constraint(posY + crop_half, 0, height)
    imgCropped[xMinFixed:xMaxFixed, yMinFixed:yMaxFixed] = \
        img[pxMin:pxMax + 1, pyMin:pyMax + 1]

    return imgCropped


class Blob:
    # posX: x coordinate (0-184)
    # posY: y coordinate (0-104)
    # ID: blob ID (0-11)
    # force: force applied to the blob (0-?)
    # area: area of the blob
    # t_appeared: timestamp of the appeared time
    # points: coordinates of blob pixels

    def __init__(self, cx, cy, area, force, points, contour, sub_cx, sub_cy):
        self.cx = cx
        self.cy = cy
        self.c = (cx, cy)

        self.sub_cx = sub_cx
        self.sub_cy = sub_cy
        self.sub_c = (sub_cx, sub_cy)

        # self.ID = ID
        self.force = force
        self.area = area

        self.t_appeared = time.time()
        self.points = points
        self.contour = contour
        self.lifetime = 0

        self.slot = -1
        self.phase = 0

    def update(self, blob):
        self.cx = blob.cx
        self.cy = blob.cy
        self.c = (blob.cx, blob.cy)

        self.sub_cx = blob.sub_cx
        self.sub_cy = blob.sub_cy
        self.sub_c = (blob.sub_cx, blob.sub_cy)

        # self.ID = ID
        self.force = blob.force
        self.area = blob.area

        self.points = blob.points
        self.contour = blob.contour
        self.lifetime = 0

    def attributeID(self, ID):
        self.ID = ID
        # print('attributed ID: %d' % ID)

    def succeedTime(self, time):
        self.t_appeared = time


def detectBlobs(img, areaThreshold=1000, forceThreshold=10, binThreshold=2, interp=5):

    contours = []
    hierarchy = []
    # moments = []
    areas = []
    cxs = []
    cys = []
    # pixelpoints = []
    forces = []
    blobs = []
    img_thre = np.zeros_like(img)
    img_interp = np.zeros((img.shape[0] * interp, img.shape[1] * interp))

    if np.max(img) > 0:
        # img_uint8 = np.zeros_like(img, dtype=np.uint8)
        # img_uint8 = (img / np.max(img) * 255).astype(np.uint8)
        img_thre = copy.deepcopy(img) * 2
        img_thre[img_thre >= 255] = 255
        img_thre = img_thre.astype(np.uint8)

        # Binary threshold
        img_thre = cv2.threshold(img_thre,
                                 binThreshold,
                                 255,
                                 cv2.THRESH_BINARY)[1]
        # find contours
        contours, hierarchy = cv2.findContours(
            img_thre,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # remove peaks which are included in large blobs
        # make masks for blobs over area threshold
        mask = np.zeros(img.shape, dtype=np.uint8)
        for cnt in contours:
            # print(cnt)
            area = cv2.contourArea(cnt)
            # print(area)
            if area > areaThreshold:
                cv2.drawContours(mask, [cnt], 0, 255, -1)
        img_masked = cv2.subtract(img, mask.astype(np.float64))

        # find peaks
        # img_peaks = findLocalPeaks(img, threshold=0.5, kernal=5).astype(np.uint8)

        # find peak from interpolated image
        img_interp = cv2.resize(img_masked, (img.shape[1] * interp, img.shape[0] * interp), interpolation=cv2.INTER_LANCZOS4)
        img_peaks_interp = findLocalPeaks(img_interp, threshold=2, kernal=5 * interp)

        img_thre = mask

        # extract coordinates from peak image
        # peaks = findPeakCoord(img_peaks)
        # sub_peaks = findSubpixelPeaks(peaks, img, n=7)
        sub_peaks = findPeakCoord(img_peaks_interp)

        for peak in sub_peaks:

            cx = int(peak[1] / interp)
            cy = int(peak[0] / interp)
            # peak coordinates
            cxs.append(peak[1] / interp)
            cys.append(peak[0] / interp)

            # force calculation
            cropped = cropImage(img, (cy, cx), 1, margin=0)
            if np.shape(cropped)[0] == 0 or np.shape(cropped)[0] == 0:
                force = 0
            else:
                # calculate force from the raw input image
                force = np.sum(cropped)
            forces.append(force)

            # create blob objects
            b = Blob(cx, cy, 3 * 3, force, [], [], peak[1] / interp, peak[0] / interp)
            # if b.area < areaThreshold:
            if b.force > forceThreshold:
                blobs.append(b)
        '''
        for cnt in contours:
            # moment
            M = cv2.moments(cnt)
            moments.append(M)

            # area
            area = cv2.contourArea(cnt)
            areas.append(area)

            # centroid
            try:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            except ZeroDivisionError:
                # print('zero division error!')
                # calculate center by four extreme points
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
                cx = (leftmost[0] + rightmost[0] + topmost[0] + bottommost[0]) / 2
                cy = (leftmost[1] + rightmost[1] + topmost[1] + bottommost[1]) / 2
            cxs.append(cx)
            cys.append(cy)

            # extract blob mask for exact force calculation
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            # pixelpoint = np.transpose(np.nonzero(mask))
            pixelpoint = np.nonzero(mask)
            pixelpoints.append(pixelpoint)

            # calculate force from the raw input image
            force = np.sum(img[pixelpoint])
            forces.append(force)

            # create blob objects
            b = Blob(cx, cy, area, force, pixelpoint, cnt)
            if b.area < areaThreshold:
                if b.force > forceThreshold:
                    blobs.append(b)
        '''

        # print(pixelpoints)

    # print(contours)
    return blobs, contours, hierarchy, areas, cxs, cys, forces, img_thre, img_interp


class TrackBlobs():
    def __init__(self, interp=5):
        # set initial parameters
        self.nextID = 0
        self.prevBlobs = []
        self.interp = interp
        # self.IDTable = [False] * 1000

    # def registerID(self, blob):
    #     # .index returns the index of the first item appears in the list
    #     availableID = self.IDTable.index(False)
    #     # print(self.IDTable)
    #     print(availableID)
    #     blob.attributeID(availableID)
    #     self.IDTable[availableID] = True

    #     return blob

    def update(self, img):
        # find blobs in current frame
        self.currentBlobs = detectBlobs(img, areaThreshold=1000, interp=self.interp)[0]

        # no blobs in the image
        if len(self.currentBlobs) == 0:
            # reset next ID
            self.nextID = 0
            # self.IDTable = [False] * 1000
            self.prevBlobs = []
            return self.currentBlobs

        # prepare distance matrix
        # current centroids
        currentCentroids = np.zeros((len(self.currentBlobs), 2), dtype=np.float)
        previousCentroids = np.zeros((len(self.prevBlobs), 2), dtype=np.float)

        # store blob coordinates
        for i in range(len(self.currentBlobs)):
            currentCentroids[i] = self.currentBlobs[i].c
        for i in range(len(self.prevBlobs)):
            previousCentroids[i] = self.prevBlobs[i].c

        # if there are no blobs being tracked, register all current blobs
        if len(self.prevBlobs) == 0:
            for b in self.currentBlobs:
                # print(b.c)
                # b = self.registerID(b)
                b.attributeID(self.nextID)
                # print('no prev blobs!')
                self.nextID += 1
        else:
            # calculate distance of all pairs of current and previous blobs
            distMat = dist.cdist(
                previousCentroids,
                currentCentroids,
                metric='euclidean'
            )

            # print(distMat)

            # sort the matrix by element's min values
            rows = distMat.min(axis=1).argsort()
            cols = distMat.argmin(axis=1)[rows]
            # print(rows)
            # print(cols)

            # check if the combination is already used
            usedRows = set()
            usedCols = set()

            # iterate over row, columns
            for (row, col) in zip(rows, cols):
                # ignore already examined rows, cols.
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, update ID of the current blobs with
                # previous blob IDs, and maintain appeared time.
                blobID = self.prevBlobs[row].ID
                blobTime = self.prevBlobs[row].t_appeared
                self.currentBlobs[col].attributeID(blobID)
                # print('ID updated!')
                self.currentBlobs[col].succeedTime(blobTime)

                # check that we have examined the row, col
                usedRows.add(row)
                usedCols.add(col)

            # extract unchecked rows, cols
            # unusedRows = set(range(0, distMat.shape[0])).difference(usedRows)
            unusedCols = set(range(0, distMat.shape[1])).difference(usedCols)

            # print('unused rows: ', unusedRows)
            # print('unused cols: ', unusedCols)
            # if the number of prev blobs are greater than or equal to
            # current blobs, check their liftime
            for col in unusedCols:
                # self.register(inputCentroids[col])
                self.currentBlobs[col].attributeID(self.nextID)
                self.nextID += 1

        # toss the current blob information to prev
        self.prevBlobs = self.currentBlobs

        return self.currentBlobs


class marker:
    # pos_x: x coordinate (0-184)
    # delta pos_x
    # pos_y: y coordinate (0-104)
    # delta pos_y
    # timestamp: time when the marker first created
    # rotation: orientation of marker (0-2pi)
    # delta rotation

    def __init__(self, blobs, img, marker_size, magnet_dist, magnet_offset, interpolation=3):

        # static magnet blobs
        # self.blobs = [None] * 4
        self.blobs = blobs

        # marker size
        self.marker_size = marker_size
        self.magnet_dist = magnet_dist
        self.magnet_offset = magnet_offset
        self.magnet_dist_short = self.magnet_dist - self.magnet_offset

        # initial position
        self.pos = (0, 0)
        self.calculateMarkerCenter()
        self.pos_x = self.pos[0]
        self.pos_y = self.pos[1]
        self.d_pos_x = 0
        self.d_pos_y = 0

        self.timestamp = time.time()

        self.rot = (0, 0)
        self.rot_rad = np.arctan2(self.rot[1], self.rot[0])
        self.calculateRotation()
        self.d_rot = (0, 0)

        self.desired_blob_pos = [None] * 4
        self.calculateDesiredBlobPos()

        self.marker_border = [None] * 4
        self.calculateMarkerBorder()

        self.img = img
        self.crop_img = img

        self.pressure_values = np.zeros((5, 3, 3), dtype=np.float)

        self.itp = interpolation
        self.sub_region = np.zeros((5 * 3 * self.itp, 5 * 3 * self.itp))

        self.lifetime = 0

    def update(self, blobs, img):
        '''
        update marker information
        '''
        # update blob (magnet) positions
        for i in range(4):
            # check if the magnet blob still exists
            if self.blobs[i] is not None:
                b_exist = self.blobs[i]
                isExist = False
                for b in blobs:
                    if b_exist.ID is b.ID:
                        self.blobs[i] = b
                        isExist = True
                if not isExist:
                    self.blobs[i] = None

        # update center coordinate
        prev_pos = self.pos
        self.calculateMarkerCenter()
        self.pos_x = self.pos[0]
        self.pos_y = self.pos[1]

        # update rotation
        prev_rot = self.rot
        self.calculateRotation()
        self.rot_rad = np.arctan2(self.rot[1], self.rot[0])

        # calculate desired position of magnet
        self.calculateDesiredBlobPos()

        # calculate marker border
        self.calculateMarkerBorder()

        # crop marker img
        self.img = img
        self.cropMarkerRegion()

        # get pressure values
        self.getPressureValues()

        # if the blob is gone, find the blob from the desired coords
        for i in range(4):
            if self.blobs[i] is None:
                for b in blobs:
                    dist = distance(b.sub_c, self.desired_blob_pos[i])
                    if dist < 3:
                        self.blobs[i] = b

        # update deltas
        self.d_pos_x = self.pos_x - prev_pos[0]
        self.d_pos_y = self.pos_y - prev_pos[1]
        self.d_pos = (self.d_pos_x, self.d_pos_y)

        self.d_rot = (self.rot[0] - prev_rot[0], self.rot[1] - prev_rot[1])

    def calculateDesiredBlobPos(self):
        '''
        calculate desired magnet blob position from pos and rot
        '''

        # p0
        pos_temp = [0, 0]
        pos_temp[0] = self.pos[0] + self.magnet_dist / 2 * np.sqrt(2) * np.cos(self.rot_rad + np.pi * 3 / 4)
        pos_temp[1] = self.pos[1] + self.magnet_dist / 2 * np.sqrt(2) * np.sin(self.rot_rad + np.pi * 3 / 4)
        self.desired_blob_pos[0] = pos_temp

        # p1
        pos_temp = [0, 0]
        pos_temp[0] = self.pos[0] + self.magnet_dist / 2 * np.sqrt(2) * np.cos(self.rot_rad + np.pi * 1 / 4)
        pos_temp[1] = self.pos[1] + self.magnet_dist / 2 * np.sqrt(2) * np.sin(self.rot_rad + np.pi * 1 / 4)
        self.desired_blob_pos[1] = pos_temp

        # p2
        pos_temp = [0, 0]
        theta = np.arctan2(self.magnet_dist_short - self.magnet_dist / 2, self.magnet_dist / 2)
        mag = np.sqrt((self.magnet_dist_short - self.magnet_dist / 2) ** 2 + (self.magnet_dist / 2) ** 2)
        pos_temp[0] = self.pos[0] + mag * np.cos(self.rot_rad - theta)
        pos_temp[1] = self.pos[1] + mag * np.sin(self.rot_rad - theta)
        self.desired_blob_pos[2] = pos_temp

        # p3
        pos_temp = [0, 0]
        pos_temp[0] = self.pos[0] + self.magnet_dist / 2 * np.sqrt(2) * np.cos(self.rot_rad - np.pi * 3 / 4)
        pos_temp[1] = self.pos[1] + self.magnet_dist / 2 * np.sqrt(2) * np.sin(self.rot_rad - np.pi * 3 / 4)
        self.desired_blob_pos[3] = pos_temp

    def calculateMarkerBorder(self):
        '''
        calculate borderline of marker
        p2 o----------o p1
           |          |
           |          |
           |          |
        p3 o----------o p0
        '''

        # p0
        pos_temp = [0, 0]
        pos_temp[0] = self.pos[0] + self.marker_size / 2 * np.sqrt(2) * np.cos(self.rot_rad + np.pi * 3 / 4)
        pos_temp[1] = self.pos[1] + self.marker_size / 2 * np.sqrt(2) * np.sin(self.rot_rad + np.pi * 3 / 4)
        self.marker_border[0] = pos_temp

        # p1
        pos_temp = [0, 0]
        pos_temp[0] = self.pos[0] + self.marker_size / 2 * np.sqrt(2) * np.cos(self.rot_rad + np.pi * 1 / 4)
        pos_temp[1] = self.pos[1] + self.marker_size / 2 * np.sqrt(2) * np.sin(self.rot_rad + np.pi * 1 / 4)
        self.marker_border[1] = pos_temp

        # p2
        pos_temp = [0, 0]
        pos_temp[0] = self.pos[0] + self.marker_size / 2 * np.sqrt(2) * np.cos(self.rot_rad - np.pi * 1 / 4)
        pos_temp[1] = self.pos[1] + self.marker_size / 2 * np.sqrt(2) * np.sin(self.rot_rad - np.pi * 1 / 4)
        self.marker_border[2] = pos_temp

        # p3
        pos_temp = [0, 0]
        pos_temp[0] = self.pos[0] + self.marker_size / 2 * np.sqrt(2) * np.cos(self.rot_rad - np.pi * 3 / 4)
        pos_temp[1] = self.pos[1] + self.marker_size / 2 * np.sqrt(2) * np.sin(self.rot_rad - np.pi * 3 / 4)
        self.marker_border[3] = pos_temp

    def cropMarkerRegion(self):
        '''
        perform Affine transform to crop marker image for pressure extraction
        '''
        # pts are in counterclockwise
        pts1 = np.float32(
            [list(self.marker_border[1]), list(self.marker_border[0]), list(self.marker_border[3])]
        )
        pts2 = np.float32(
            [
                [self.marker_size * self.itp - 1, 0],
                [self.marker_size * self.itp - 1, self.marker_size * self.itp - 1],
                [0, self.marker_size * self.itp - 1]
            ]
        )

        # create rotation matrix
        M = cv2.getAffineTransform(pts1, pts2)

        # rotate marker region
        self.crop_img = cv2.warpAffine(self.img, M, (self.marker_size * self.itp, self.marker_size * self.itp), flags=cv2.INTER_LANCZOS4)

    def getPressureValues(self):
        '''
        get pressure values from cropped image
               ^
               |
             -----
             | 1 |
         ----|   |----
         | 4   0   2 |
         ----|   |----
             | 3 |
             -----
        '''
        # crop subregion first
        sub_region_0 = np.rot90(self.crop_img, 0)[5 * 3 * self.itp:5 * 6 * self.itp + 1, 5 * 3 * self.itp:5 * 6 * self.itp + 1]
        sub_region_1 = np.rot90(self.crop_img, 2)[5 * 6 * self.itp:5 * 9 * self.itp + 1, 5 * 3 * self.itp:5 * 6 * self.itp + 1]
        sub_region_2 = np.rot90(self.crop_img, 3)[5 * 6 * self.itp:5 * 9 * self.itp + 1, 5 * 3 * self.itp:5 * 6 * self.itp + 1]
        sub_region_3 = np.rot90(self.crop_img, 0)[5 * 6 * self.itp:5 * 9 * self.itp + 1, 5 * 3 * self.itp:5 * 6 * self.itp + 1]
        sub_region_4 = np.rot90(self.crop_img, 1)[5 * 6 * self.itp:5 * 9 * self.itp + 1, 5 * 3 * self.itp:5 * 6 * self.itp + 1]

        self.sub_region = sub_region_0
        sub_region = [sub_region_0, sub_region_1, sub_region_2, sub_region_3, sub_region_4]
        # region 0

        for sub_r in range(len(sub_region)):
            for i in range(3):
                for j in range(3):
                    # print(sub_r, i, j)
                    self.pressure_values[sub_r][i][j] = np.sum(sub_region[sub_r][(i * 5 + 1) * self.itp:((i + 1) * 5 - 1) * self.itp + 1,
                                                                                 (j * 5 + 1) * self.itp:((j + 1) * 5 - 1) * self.itp + 1])

    def calculateMarkerCenter(self):
        '''
        calculate marker center from magnet points
        '''
        if sum(1 for _ in filter(None.__ne__, self.blobs)) >= 2:
            count = 0
            pos = [0, 0]
            if self.blobs[0] is not None and\
               self.blobs[1] is not None:
                count += 1
                v_temp = vector(self.blobs[0], self.blobs[1])
                pos_x = self.blobs[0].sub_cx + v_temp.unit_vector()[0] * v_temp.length() / 2 + self.magnet_dist * np.cos(v_temp.angle() - np.pi / 2) / 2
                pos_y = self.blobs[0].sub_cy + v_temp.unit_vector()[1] * v_temp.length() / 2 + self.magnet_dist * np.sin(v_temp.angle() - np.pi / 2) / 2
                # print('pos from p0 p1:', (pos_x, pos_y))
                pos[0] += pos_x
                pos[1] += pos_y

            if self.blobs[0] is not None and\
               self.blobs[2] is not None:
                count += 1
                v_temp = vector(self.blobs[0], self.blobs[2])
                v_temp.rotate(-(np.arctan2(1, 1) - np.arctan2(self.magnet_dist_short, self.magnet_dist)))
                v_temp.vx *= np.sqrt(self.magnet_dist ** 2 * 2) / np.sqrt(self.magnet_dist ** 2 + self.magnet_dist_short ** 2)
                v_temp.vy *= np.sqrt(self.magnet_dist ** 2 * 2) / np.sqrt(self.magnet_dist ** 2 + self.magnet_dist_short ** 2)
                pos_x = self.blobs[0].sub_cx + v_temp.unit_vector()[0] * v_temp.length() / 2
                pos_y = self.blobs[0].sub_cy + v_temp.unit_vector()[1] * v_temp.length() / 2
                # print('pos from p0 p2:', (pos_x, pos_y))
                pos[0] += pos_x
                pos[1] += pos_y

            if self.blobs[0] is not None and\
               self.blobs[3] is not None:
                count += 1
                v_temp = vector(self.blobs[0], self.blobs[3])
                pos_x = self.blobs[0].sub_cx + v_temp.unit_vector()[0] * v_temp.length() / 2 + self.magnet_dist * np.cos(v_temp.angle() + np.pi / 2) / 2
                pos_y = self.blobs[0].sub_cy + v_temp.unit_vector()[1] * v_temp.length() / 2 + self.magnet_dist * np.sin(v_temp.angle() + np.pi / 2) / 2
                # print('pos from p0 p3:', (pos_x, pos_y))
                pos[0] += pos_x
                pos[1] += pos_y

            if self.blobs[1] is not None and\
               self.blobs[2] is not None:
                count += 1
                v_temp = vector(self.blobs[1], self.blobs[2])
                pos_x = self.blobs[1].sub_cx + v_temp.unit_vector()[0] * v_temp.length() / 2 * self.magnet_dist / self.magnet_dist_short + self.magnet_dist * np.cos(v_temp.angle() - np.pi / 2) / 2
                pos_y = self.blobs[1].sub_cy + v_temp.unit_vector()[1] * v_temp.length() / 2 * self.magnet_dist / self.magnet_dist_short + self.magnet_dist * np.sin(v_temp.angle() - np.pi / 2) / 2
                # print('pos from p1 p2:', (pos_x, pos_y))
                pos[0] += pos_x
                pos[1] += pos_y

            if self.blobs[1] is not None and\
               self.blobs[3] is not None:
                count += 1
                pos_temp = ((self.blobs[1].sub_cx + self.blobs[3].sub_cx) / 2, (self.blobs[1].sub_cy + self.blobs[3].sub_cy) / 2)
                pos_x = pos_temp[0]
                pos_y = pos_temp[1]
                # print('pos from p1 p3:', (pos_x, pos_y))
                pos[0] += pos_x
                pos[1] += pos_y

            if self.blobs[3] is not None and\
               self.blobs[2] is not None:
                count += 1
                v_temp = vector(self.blobs[3], self.blobs[2])
                v_temp.rotate(-np.arctan2(self.magnet_offset, self.magnet_dist))
                v_temp.vx *= self.magnet_dist / np.sqrt(self.magnet_dist ** 2 + self.magnet_offset ** 2)
                v_temp.vy *= self.magnet_dist / np.sqrt(self.magnet_dist ** 2 + self.magnet_offset ** 2)
                pos_x = self.blobs[3].sub_cx + v_temp.unit_vector()[0] * v_temp.length() / 2 + self.magnet_dist * np.cos(v_temp.angle() + np.pi / 2) / 2
                pos_y = self.blobs[3].sub_cy + v_temp.unit_vector()[1] * v_temp.length() / 2 + self.magnet_dist * np.sin(v_temp.angle() + np.pi / 2) / 2
                # print('pos from p3 p2:', (pos_x, pos_y))
                pos[0] += pos_x
                pos[1] += pos_y

            # print('count:  ', count)
            pos[0] /= count
            pos[1] /= count

            # print('pos sum:', pos)
            self.pos = pos

        else:
            self.pos = self.pos

    def calculateRotation(self):
        '''
        calculate rotation using available marker points
        '''

        if sum(1 for _ in filter(None.__ne__, self.blobs)) >= 2:
            count = 0
            v_rot = [0, 0]
            if self.blobs[0] is not None and\
               self.blobs[1] is not None:
                count += 1
                v_temp = vector(self.blobs[0], self.blobs[1])
                [vx, vy] = v_temp.unit_vector()
                # print('vx, vy:', [vx, vy])
                v_rot[0] += vx
                v_rot[1] += vy

            if self.blobs[1] is not None and\
               self.blobs[2] is not None:
                count += 1
                v_temp = vector(self.blobs[1], self.blobs[2])
                v_temp.rotate(np.pi / 2)
                [vx, vy] = v_temp.unit_vector()
                # print('vx, vy:', [vx, vy])
                v_rot[0] += vx
                v_rot[1] += vy

            if self.blobs[0] is not None and\
               self.blobs[3] is not None:
                count += 1
                v_temp = vector(self.blobs[0], self.blobs[3])
                v_temp.rotate(np.pi / 2)
                [vx, vy] = v_temp.unit_vector()
                # print('vx, vy:', [vx, vy])
                v_rot[0] += vx
                v_rot[1] += vy

            if self.blobs[3] is not None and\
               self.blobs[2] is not None:
                count += 1
                v_temp = vector(self.blobs[3], self.blobs[2])
                v_temp.rotate(-np.arctan2(self.magnet_offset, self.magnet_dist))
                [vx, vy] = v_temp.unit_vector()
                # print('vx, vy:', [vx, vy])
                v_rot[0] += vx
                v_rot[1] += vy

            if self.blobs[0] is not None and\
               self.blobs[2] is not None:
                count += 1
                v_temp = vector(self.blobs[0], self.blobs[2])
                v_temp.rotate(np.arctan2(self.magnet_dist_short, self.magnet_dist))
                [vx, vy] = v_temp.unit_vector()
                # print('vx, vy:', [vx, vy])
                v_rot[0] += vx
                v_rot[1] += vy

            if self.blobs[3] is not None and\
               self.blobs[1] is not None:
                count += 1
                v_temp = vector(self.blobs[3], self.blobs[1])
                v_temp.rotate(-np.pi / 4)
                [vx, vy] = v_temp.unit_vector()
                # print('vx, vy:', [vx, vy])
                v_rot[0] += vx
                v_rot[1] += vy

            v_rot[0] /= count
            v_rot[1] /= count

            self.rot = v_rot
        else:
            self.rot = self.rot

    def isBlobinMarker(self, blob):
        '''
        check if the blob is inside the marker region
        '''

        # marker area
        marker_area = self.marker_size * self.marker_size

        # area of triangle made from the blob and marker borders
        A1 = blob.sub_cx * (self.marker_border[0][1] - self.marker_border[1][1]) +\
            self.marker_border[0][0] * (self.marker_border[1][1] - blob.sub_cy) +\
            self.marker_border[1][0] * (blob.sub_cy - self.marker_border[0][1])
        A2 = blob.sub_cx * (self.marker_border[1][1] - self.marker_border[2][1]) +\
            self.marker_border[1][0] * (self.marker_border[2][1] - blob.sub_cy) +\
            self.marker_border[2][0] * (blob.sub_cy - self.marker_border[1][1])
        A3 = blob.sub_cx * (self.marker_border[2][1] - self.marker_border[3][1]) +\
            self.marker_border[2][0] * (self.marker_border[3][1] - blob.sub_cy) +\
            self.marker_border[3][0] * (blob.sub_cy - self.marker_border[2][1])
        A4 = blob.sub_cx * (self.marker_border[3][1] - self.marker_border[0][1]) +\
            self.marker_border[3][0] * (self.marker_border[0][1] - blob.sub_cy) +\
            self.marker_border[0][0] * (blob.sub_cy - self.marker_border[3][1])

        A_sum = A1 + A2 + A3 + A4

        if A_sum < marker_area:
            return True
        else:
            return False


class vector:
    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2

        self.vx = b2.sub_cx - b1.sub_cx
        self.vy = b2.sub_cy - b1.sub_cy

    def length(self):
        # return np.sqrt((self.b1.sub_cx - self.b2.sub_cx) ** 2 + (self.b1.sub_cy - self.b2.sub_cy) ** 2)
        return np.sqrt(self.vx ** 2 + self.vy ** 2)

    def angle(self):
        return np.arctan2(self.vy, self.vx)

    def unit_vector(self):
        return (self.vx / self.length(), self.vy / self.length())

    def rotate(self, rad):
        temp_vx = self.vx
        temp_vy = self.vy
        self.vx = temp_vx * np.cos(rad) - temp_vy * np.sin(rad)
        self.vy = temp_vx * np.sin(rad) + temp_vy * np.cos(rad)


def vec_angle(v1, v2):
    return np.arccos((v1.vx * v2.vx + v1.vy * v2.vy) / (v1.length() * v2.length()))


# TODO
# crop marker region
# define grid and extract sensing blobs
class TrackMarkers():
    def __init__(self, marker_size, magnet_dist, magnet_offset):
        # set initial parameters
        self.markers = []

        self.marker_size = marker_size
        self.magnet_dist = magnet_dist
        self.magnet_offset = magnet_offset
        self.magnet_dist_short = self.magnet_dist - self.magnet_offset
        # self.distanceTolerance = 1

        self.t_threshold = 1

    def findMarkerBlobs(self, blobs):
        '''
        marker shape
        p2     o-----o p1
              /      |
             /       | v1
            /        |
        p3 o---------o p0
                v2

        return: p0, p1, p2, p3
        '''
        points = []

        if len(blobs) > 3:  # while there are more than 3 blobs
            # for a blob
            for b1 in blobs:
                vectors = []
                distances = []
                for b2 in blobs:
                    # validate distance then make vectors
                    distance = np.sqrt((b1.sub_cx - b2.sub_cx) ** 2 + (b1.sub_cy - b2.sub_cy) ** 2)
                    if distance > self.magnet_dist - 4 and distance < self.magnet_dist + 2:
                        distances.append(distance)
                        vectors.append(vector(b1, b2))

                # find marker point 'p0'
                if len(distances) >= 2:
                    for v_pair in itertools.combinations(vectors, 2):
                        v_angle = vec_angle(v_pair[0], v_pair[1])
                        len_diff = np.abs(v_pair[0].length() - v_pair[1].length())
                        # print('vector angle: %f' % np.rad2deg(vec_angle(v_pair[0], v_pair[1])))
                        if v_pair[0].length() > self.magnet_dist - 1 and\
                           v_pair[0].length() < self.magnet_dist + 1 and\
                           v_pair[1].length() > self.magnet_dist - 1 and\
                           v_pair[1].length() < self.magnet_dist + 1 and\
                           len_diff < 1 and\
                           np.rad2deg(v_angle) > 88 and\
                           np.rad2deg(v_angle) < 92:
                            # determine which vector is v1
                            p_temp = (v_pair[0].b1.sub_cx + (v_pair[0].b2.sub_cx - v_pair[0].b1.sub_cx) * np.cos(-np.pi / 2) - (v_pair[0].b2.sub_cy - v_pair[0].b1.sub_cy) * np.sin(-np.pi / 2),
                                      v_pair[0].b1.sub_cy + (v_pair[0].b2.sub_cx - v_pair[0].b1.sub_cx) * np.sin(-np.pi / 2) + (v_pair[0].b2.sub_cy - v_pair[0].b1.sub_cy) * np.cos(-np.pi / 2))

                            # check if the temp point fits with another vector point
                            distance = np.sqrt((v_pair[1].b2.sub_cx - p_temp[0]) ** 2 + (v_pair[1].b2.sub_cy - p_temp[1]) ** 2)
                            # print(distance)
                            if distance < 2:
                                v1 = v_pair[0]
                                v2 = v_pair[1]
                            else:
                                v1 = v_pair[1]
                                v2 = v_pair[0]

                            # find diagonal point 'p2'
                            # draw parallel line to 'v2' with length of 32
                            v2_u = (v2.b2.sub_cx - v2.b1.sub_cx, v2.b2.sub_cy - v2.b1.sub_cy) / v2.length()
                            p2_candidate = (v1.b2.sub_cx + v2_u[0] * (self.magnet_dist_short),
                                            v1.b2.sub_cy + v2_u[1] * (self.magnet_dist_short))

                            # check if the 'p2' is there
                            for b in blobs:
                                distance = np.sqrt((b.sub_cx - p2_candidate[0]) ** 2 + (b.sub_cy - p2_candidate[1]) ** 2)
                                # print(distance)
                                if distance < 1:
                                    # save points
                                    p0 = v1.b1
                                    p1 = v1.b2
                                    p2 = b
                                    p3 = v2.b2

                                    points = [p0, p1, p2, p3]
                                    # temp_marker.update([p0, p1, p2, p3])
                                    blobs_unused = [blob for blob in blobs if blob not in points]
                                    # return temp_marker, blobs_unused
                                    return points, blobs_unused
        return points, blobs

    def findMarker(self, blobs, img):
        # for combination of two blobs, find circle center
        # for the circle center, calculate distance from any other blobs
        # if there are at least 7 blobs with matching distance, confirm it as a center

        markers = []

        while len(blobs) > 3:  # while there are more than 3 blobs
            marker_blobs, blobs = self.findMarkerBlobs(blobs)
            # print(marker, blobs)
            if len(marker_blobs) == 0:
                break
            else:
                temp_marker = marker(marker_blobs, img, self.marker_size, self.magnet_dist, self.magnet_offset)
                # print('temp_marker blobs:', temp_marker.blobs)
                markers.append(temp_marker)
                # return markers, blobs

        return markers, blobs

    def update(self, blobs, img):
        # print('blobs: ', blobs)
        # print('markers: ', self.markers)
        # for existing markers, update their information and exclude the marker's blobs from current blobs
        blobs_mkr = []
        for mkr in self.markers:
            # print(mkr)
            # print('marker blobs: ', mkr.blobs)
            mkr.update(blobs, img)
            for b in mkr.blobs:
                blobs_mkr.append(b)
        self.blobs_unused = [blob for blob in blobs if blob not in blobs_mkr]
        # print(self.blobs_unused)

        # for unused blobs, determine if they belong to any markers
        blob_mkr = []
        for b in self.blobs_unused:
            for mkr in self.markers:
                if mkr.isBlobinMarker(b):
                    blob_mkr.append(b)

        self.blobs_unused = [blob for blob in self.blobs_unused if blob not in blobs_mkr]

        # for recent blobs, find markers
        self.t_current = time.time()

        # filter recent blobs
        # self.blobs = blobs
        self.recent_blobs = []
        for b in self.blobs_unused:
            if self.t_current - b.t_appeared < self.t_threshold:
                self.recent_blobs.append(b)

        # find markers
        # print('find markers')
        temp_markers = []
        new_markers = []
        if len(self.recent_blobs) > 3:
            new_markers, blobs_unused = self.findMarker(self.recent_blobs, img)

        # print('new markers:', new_markers)
        # check for existing markers
        if len(self.markers) is 0:
            # print('initial marker!')
            for mkr in new_markers:
                temp_markers.append(mkr)
                break
        else:
            # print('check existing markers!')
            # check existence of marker
            for mkr_exist in self.markers:
                if mkr_exist.blobs.count(None) == 4:
                    mkr_exist.lifetime += 1
                    if mkr_exist.lifetime < 20:
                        temp_markers.append(mkr_exist)
                else:
                    mkr_exist.lifetime = 0
                    temp_markers.append(mkr_exist)

            for mkr in new_markers:
                isExist = False
                for mkr_exist in self.markers:
                    if distance(mkr.pos, mkr_exist.pos) < 15:
                        # print('existing marker!')
                        # mkr_exist.update(img)
                        # temp_markers.append(mkr_exist)
                        isExist = True
                        break

                if not isExist:
                    # print('new marker!')
                    temp_markers.append(mkr)

        self.markers = temp_markers
