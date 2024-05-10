import numpy as np
import cv2 as cv
import math
import os
import sys


class Args(object):
    image = "/home/charly/Videos/Github/stable/video/vid1.mp4"
    analysis = None

def smooth(data,num_analyzed_frames,samples_per_second,freq_cutoff = 0.2):

    N = num_analyzed_frames * 2 - 2
    f = np.fft.fftfreq(N) * samples_per_second
    fft = np.fft.fft(data)
    fft[np.abs(f) > freq_cutoff] = 0
    smoothed = np.real(np.fft.ifft(fft))

    return smoothed
def resize_to_fit(frame, size,stable=False):
    desired_width, desired_height = size
    actual_height, actual_width = tuple(frame.shape[:2])
    factor = min(desired_width / actual_width, desired_height / actual_height)
    if stable:
        factor = desired_width / actual_width
        return cv.resize(frame, (0,0), fx=factor, fy=factor)
    else:
        return cv.resize(frame, (0,0), fx=factor, fy=factor)

args = Args()

if args.analysis is None:
    args.analysis = os.path.splitext(args.image)[0] + '.analysis'


if not os.path.exists(args.analysis):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 107,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it

    cap = cv.VideoCapture(args.image)
    ret, old_frame = cap.read()
    frame_height, frame_width = tuple(old_frame.shape[:2])
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


    cv.namedWindow("frame")
    cv.moveWindow("frame", 20, 20);

    min_diff_x = 0
    min_diff_y = 0
    max_diff_x = 0
    max_diff_y = 0

    current_loc_x = 0
    current_loc_y = 0

    frame_locations = []

    # Create a mask image for drawing purposes
    while(1):
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        if good_new.size == 0 or good_old.size == 0:
            print('points lost')
            break

        diffsX = []
        diffsY = []

        # draw the tracks
        mask = np.zeros_like(old_frame)
        for i, (new,old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            diffsX.append(c - a)
            diffsY.append(d - b)

            mask = cv.line(mask, (int(a),int(b)), (int(c),int(d)), color[i].tolist(), 2)
            mask = cv.circle(mask, (int(a),int(b)), 3, color[i].tolist(), -1)

        diffX = np.median(diffsX)
        diffY = np.median(diffsY)

        current_loc_x += diffX
        current_loc_y += diffY

        frame_locations.append((current_loc_x, current_loc_y))

        min_diff_x = min(current_loc_x, min_diff_x)
        min_diff_y = min(current_loc_y, min_diff_y)
        max_diff_x = max(current_loc_x, max_diff_x)
        max_diff_y = max(current_loc_y, max_diff_y)


        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        if p0.shape[0] < 50:
            p0 = np.unique(np.concatenate((
                p0,
                cv.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            )), axis=0)[:100]


        k = cv.waitKey(25) & 0xff
        if k == 27:
            break

        num_analyzed_frames = len(frame_locations)
        samples_per_second = 30

        if num_analyzed_frames > 1:

            xs = list(map(lambda frame: frame[0], frame_locations))
            ys = list(map(lambda frame: frame[1], frame_locations))

            # take the points, mirror them and add them to the end to complete a loop
            # which is necessary for the fourier transform to work.
            xs = np.array(xs + list(reversed(xs))[1:-1])
            ys = np.array(ys + list(reversed(ys))[1:-1])

            xs_smoothed = smooth(xs, num_analyzed_frames, samples_per_second)
            ys_smoothed = smooth(ys, num_analyzed_frames, samples_per_second)

            xs = xs[:num_analyzed_frames]
            ys = ys[:num_analyzed_frames]
            xs_smoothed = xs_smoothed[:num_analyzed_frames]
            ys_smoothed = ys_smoothed[:num_analyzed_frames]

            xs_diff = xs_smoothed - xs
            ys_diff = ys_smoothed - ys

            min_diff_x = math.floor(np.min(xs_diff))
            min_diff_y = math.floor(np.min(ys_diff))
            max_diff_x = math.ceil(np.max(xs_diff))
            max_diff_y = math.ceil(np.max(ys_diff))

            # calculate output frame sizes
            landscape = frame_width > frame_height  # Cooperate with the rear to combine the source video and the stabilized video.

            cv.namedWindow('comparison')
            cv.moveWindow('comparison', 30, 30)

            top = abs(min_diff_y) + int(ys_diff[-1])
            bottom = frame_height - abs(max_diff_y) + int(ys_diff[-1])
            left = abs(min_diff_x) + int(xs_diff[-1])
            right = frame_width - abs(max_diff_x) + int(xs_diff[-1])

            stabilized = frame[top:bottom, left:right].copy()

            frame = cv.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 4)

            stabilized_resized = resize_to_fit(stabilized, (1000, 1000),landscape)
            original_resized = resize_to_fit(frame, (1000, 1000))

            comparison = np.vstack((original_resized, stabilized_resized)) if landscape else np.hstack(
                (original_resized, stabilized_resized))

            # show video
            cv.imshow('comparison', comparison)
            k = cv.waitKey(1) & 0xff
            if k == 27:
                skipped = True

    cv.destroyAllWindows()
    print('\ndone')

    sys.exit()

