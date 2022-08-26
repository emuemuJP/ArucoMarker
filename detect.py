import cv2
from cv2 import aruco
import numpy as np
import sys

# dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()

marker_length = 0.056
# camera_matrix = np.load("mtx.npy")
# distortion_coeff = np.load("dist.npy")

camera_matrix = np.array([[639.87721705,   0.        , 330.12073612],
                            [  0.        , 643.69687408, 208.61588364],
                            [  0.        ,   0.        ,   1.        ]])
distortion_coeff = np.array([ 5.66942769e-02, -6.05774927e-01, -7.42066667e-03, -3.09571466e-04, 1.92386974e+00])

capture = cv2.VideoCapture(int(sys.argv[1]))
try:
    while True:
        ret, frame = capture.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dict_aruco, parameters=parameters)

        if len(corners) > 0:
            # マーカーごとに処理
            for i, corner in enumerate(corners):
                # rvec -> rotation vector, tvec -> translation vector
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, distortion_coeff)

                # < rodoriguesからeuluerへの変換 >

                # 不要なaxisを除去
                tvec = np.squeeze(tvec)
                rvec = np.squeeze(rvec)
                # 回転ベクトルからrodoriguesへ変換
                rvec_matrix = cv2.Rodrigues(rvec)
                rvec_matrix = rvec_matrix[0] # rodoriguesから抜き出し
                # 並進ベクトルの転置
                transpose_tvec = tvec[np.newaxis, :].T
                # 合成
                proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
                # オイラー角への変換
                euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]

                print("x : " + str(tvec[0]))
                print("y : " + str(tvec[1]))
                print("z : " + str(tvec[2]))
                print("roll : " + str(euler_angle[0]))
                print("pitch: " + str(euler_angle[1]))
                print("yaw  : " + str(euler_angle[2]))

                # 可視化
                draw_pole_length = marker_length/2 # 現実での長さ[m]
                frame = cv2.drawFrameAxes(frame, camera_matrix, distortion_coeff, rvec, tvec, draw_pole_length)

        # frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    capture.release()
    cv2.destroyWindow('frame')
except KeyboardInterrupt:
    capture.release()
    cv2.destroyWindow('frame')
