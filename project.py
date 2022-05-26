import numpy as np
import cv2
import math
import os
def nothing(x):
   pass
kernel=np.ones((3,3),np.uint8)

cv2.namedWindow("color",cv2.WINDOW_NORMAL)
cv2.createTrackbar("LH","color",0,255,nothing)
cv2.createTrackbar("LS","color",0,255,nothing)
cv2.createTrackbar("LV","color",0,255,nothing)
cv2.createTrackbar("UH","color",0,255,nothing)
cv2.createTrackbar("US","color",0,255,nothing)
cv2.createTrackbar("UV","color",0,255,nothing)
def sethsv():
    cap = cv2.VideoCapture(0)
    while 1:
        _,frame=cap.read()
        cv2.rectangle(frame, (320, 320), (80, 80), (0, 0, 0), 0)
        roi=frame[80:320,80:320]
        cvt = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # cv2.imshow("roi", roi)
        l_h = cv2.getTrackbarPos("LH", "color")
        l_s = cv2.getTrackbarPos("LS", "color")

        l_v = cv2.getTrackbarPos("LV", "color")
        u_h = cv2.getTrackbarPos("UH", "color")

        u_s = cv2.getTrackbarPos("US", "color")
        u_v = cv2.getTrackbarPos("UV", "color")

        l_b = np.array([l_h, l_s, l_v], dtype=np.uint8)
        u_b = np.array([u_h, u_s, u_v], dtype=np.uint8)
        cvt = cv2.inRange(cvt, l_b, u_b)
        cvt = cv2.erode(cvt, kernel, iterations=4)
        cvt = cv2.GaussianBlur(cvt, (5, 5), 100)
        cv2.imshow("cvt", cvt)
        cv2.imshow("video", frame)
        if (cv2.waitKey(40) == 27):
            break
def gesture():
    gesturecnt=[0,0,0,0,0,0]
    cap = cv2.VideoCapture(0)
    while 1:
       _, frame=cap.read()
       frame1=frame.copy()
       cv2.rectangle(frame1,(320,320),(80,80),(0,0,0),0)
       roi=frame[80:320,80:320]
       cvt=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

       #cv2.imshow("roi", roi)
       l_h=cv2.getTrackbarPos("LH","color")
       l_s = cv2.getTrackbarPos("LS", "color")
       l_v = cv2.getTrackbarPos("LV", "color")
       u_h = cv2.getTrackbarPos("UH", "color")
       u_s = cv2.getTrackbarPos("US", "color")
       u_v = cv2.getTrackbarPos("UV", "color")

       l_b=np.array([0,21,42],dtype=np.uint8)
       u_b = np.array([193,99,255],dtype=np.uint8)
       cvt=cv2.inRange(cvt,l_b,u_b)
       cvt=cv2.erode(cvt,kernel,iterations=4)
       cvt = cv2.dilate(cvt,kernel, iterations=4)
       cvt=cv2.GaussianBlur(cvt,(9,9),100)
       cv2.imshow("cvt",cvt)
       contour,her = cv2.findContours(cvt.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
       areas=[cv2.contourArea(c) for c in contour]

       if(len(areas)!=0):
           maxi=np.argmax(areas)
           cot=contour[maxi]
           if(areas[maxi]>10000):

             x,y,w,h=cv2.boundingRect(cot)
             cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,0),0)
             hull=cv2.convexHull(cot)

             cv2.drawContours(frame1,[cot],0,[0,255,0],0)
             cv2.drawContours(frame1, [hull], 0, [0, 0,255], 0)
             hull = cv2.convexHull(cot,returnPoints=False)
             defects=cv2.convexityDefects(cot,hull)
             num=1
             for i in range(defects.shape[0]):
                 s,e,f,d=defects[i,0]

                 st=tuple(cot[s][0])
                 end = tuple(cot[e][0])
                 f = tuple(cot[f][0])

                 a = math.sqrt((end[0]-st[0])**2+(end[1]-st[1])**2)
                 b = math.sqrt((f[0] - st[0]) ** 2 + (f[1] - st[1]) ** 2)
                 c = math.sqrt((end[0] -f[0]) ** 2 + (end[1] - f[1]) ** 2)
                 angle=math.acos((b*2+c*2-a*2)/(2*b*c))**57
                 if(angle<=90):
                     num+=1
                     cv2.circle(frame1,f,1,[0,0,0],1)
                 cv2.line(frame1,st,end,[0,255,0],2)
             if(num==1):
                 gesturecnt[num]+=1
                 print(gesturecnt[num])
                 if(gesturecnt[num]>40):
                     os.system("notepad")
                     gesturecnt[num]=0
                 cv2.putText(frame1,"gesture 1",(50,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0))
             elif (num == 2):
                 gesturecnt[num] += 1
                 if (gesturecnt[num] > 40):
                     os.chdir(r"C:\Users\katiy\Desktop")
                     os.system("start chrome.exe")
                     gesturecnt[num] = 0
                 cv2.putText(frame1, "gesture 2", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
             elif (num == 3):
                 gesturecnt[num] += 1
                 if (gesturecnt[num] > 40):
                     os.system("rundll32.exe user32.dll,LockWorkStation")
                     gesturecnt[num] = 0
                 cv2.putText(frame1, "gesture 3", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
             elif (num == 4):
                 gesturecnt[num] += 1
                 cv2.putText(frame1, "gesture 4", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
             elif (num == 5):
                 gesturecnt[num] += 1
                 cv2.putText(frame1, "gesture 5", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
       cv2.imshow("video",frame1)
       if(cv2.waitKey(40)==27):
           break
gesture()