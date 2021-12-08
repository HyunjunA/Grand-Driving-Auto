import time
import cv2
import numpy as np
from grabscreen import grab_screen
from getkeys import key_check
import sys
import os
from constants import IMAGE_WIDTH,IMAGE_HEIGHT,NUM_KEYS,W_VEC,A_VEC,S_VEC,D_VEC,WA_VEC,WD_VEC,SA_VEC,SD_VEC,NK_VEC


def main():
    while True:
        orignal_img = grab_screen((0, 0, 800, 600))
        orignal_img = cv2.resize(orignal_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        img = cv2.cvtColor(orignal_img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)      # 5x5 filter
        img = cv2.Canny(img, threshold1 = 200, threshold2 = 300)

        roi = np.array([[(0,IMAGE_HEIGHT - 75),(75, 150), (325, 150), (IMAGE_WIDTH,IMAGE_HEIGHT - 75)]], dtype=np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, roi, 255)
        masked = cv2.bitwise_and(img, mask)

        lines = cv2.HoughLinesP(masked, 2, np.pi/180, 15,np.array([]), 20, 15)

        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(masked, (x1, y1), (x2, y2), (255, 255, 255), 10)

        except:
            pass


        cv2.imshow("frame", masked)

        if cv2.waitKey(1) & 0Xff == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__' :

    main()
