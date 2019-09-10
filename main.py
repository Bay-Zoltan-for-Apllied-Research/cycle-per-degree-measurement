from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
from ar_markers import detect_markers
import cv2
import numpy as np
from math import atan2, degrees
import os
import copy
import platform
from PIL import Image
import screeninfo

#beszedes nev
def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image



def generateRectangle(size,x,y,rot):

    num = 0
    offset = 100
    a = offset
    img = np.zeros((x, y, 3), np.uint8)
    while a < y-offset:

        cv2.rectangle(img, (a, offset), (a+size, x - offset), (255, 255, 255), -1)
        a = a + size
        cv2.rectangle(img, (a, offset), (a + size, x - offset), (0, 0, 0), -1)
        a = a + size
        num = num + 1
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!MEKKOR LEGYEN????
        #minnel nagyobb a szorzo annal ritkabbak a vonalak
    rows,cols,_ = img.shape
    #vonalak forgatasa rot szoggel
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
    img = cv2.warpAffine(img,M,(cols,rows))
    return img,num


def findNearPos(inArray,number):
    number=number-1
    tmp=[]
    for i in range(1,len(inArray)):
        tmp.append(inArray[i-1]-inArray[i])

    res=[]
    x = 0
    end=False

    while(True):
        tmp1=copy.copy(tmp)

        x=x-1
        for i in range(0,len(tmp1)):
            if(tmp1[i]<x):
                tmp1[i]=0
            else:
                tmp1[i]=1
        start=-1

        for i in range(0, len(tmp1)):
            if (tmp1[i] == 1 and start == -1):
                start = i
            elif (tmp1[i] == 0 and start != -1):
                if (i - start == number):
                    res.append(start)
                    res.append(i + 1)
                    end = True
                    break
                else:
                    start = -1

        if (x<-99999):
            return -1,-1
        if(end==True):
            break
    return res
def detv2(big,num,threshParam1,threshParam2,cannyParam1,cannyParam2):
    #gray

    big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    #tresh
    ret, thresh1 = cv2.threshold(big, threshParam1,threshParam2, cv2.THRESH_BINARY)
    #canny

    edge = cv2.Canny(thresh1, cannyParam1, cannyParam2)
    #uj ures kep
    blank =  np.zeros((big.shape[0], big.shape[1], 3), np.uint8)
    #kontur kereses
    imgContours, npaContours, npaHierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #konturok sorbarendezese
    shortedContours = sorted(npaContours, key=cv2.contourArea, reverse=False)

    #kontur meret tarolasa
    shortedArea=[]
    for i in shortedContours:
        shortedArea.append(cv2.contourArea(i))
    shortedArea.append(9999999999999999999999999)
    #konturok keresese
    ret = findNearPos(shortedArea, num)
    res = []
    if(ret[0]==-1 and  ret[1]==-1):
        return size
    for i in range(ret[0], ret[1]):
        res.append(shortedArea[i])


   # cv2.waitKey(0)
    filteredContours = []
    droppedContours = []


   #avg=cv2.contourArea(shortedContours[0])
    for i in shortedContours:
        area = cv2.contourArea(i)
        if (area > shortedArea[ret[0]]/100*95 and area <shortedArea[ret[1]]/100*110):
            filteredContours.append(i)
            x, y, w, h = cv2.boundingRect(i)
            cv2.rectangle(blank, (x, y), (x + w, y + h), (0, 255, 0), 1)
        else:
            droppedContours.append(i)
    """for i in shortedContours:
        area = cv2.contourArea(i)
        if (area > cv2.contourArea(shortedContours[0])/100*40):
            filteredContours.append(i)
            x, y, w, h = cv2.boundingRect(i)
            cv2.rectangle(blank, (x, y), (x + w, y + h), (0, 255, 0), 1)
        else:
            droppedContours.append(i)
    """
    blank = cv2.drawContours(blank, filteredContours, -1, (255, 0, 0), 1)


    #data ablak rajzolasa majd ehez csinalok kulon osztalyt
    pos= 30
    blank = cv2.resize(blank, (640,480), interpolation=cv2.INTER_AREA)
    cv2.putText(blank, "marked", (pos,pos), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    edge = cv2.resize(edge, (640,480), interpolation=cv2.INTER_AREA)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
    cv2.putText(edge, "edge", (pos, pos), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    big = cv2.resize(big, (640, 480), interpolation=cv2.INTER_AREA)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2RGB)
    cv2.putText(big, "Live", (pos, pos), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB)
    thresh1 = cv2.drawContours(thresh1, droppedContours, -1, (0, 0, 255), 3)
    thresh1 = cv2.resize(thresh1, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.putText(thresh1, "thresh1", (pos, pos), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    tmp=np.hstack((edge, blank))
    tmp2 = np.hstack((big,thresh1))
    res=np.vstack((tmp,tmp2))

    text = str( "size: "+str(size)+ " num: "+ str(num)+ " len: "+ str(len(filteredContours))+ " completed: "+ str(len(filteredContours) == num))
    cv2.putText(res, text,(pos, 930), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow("data",res)
    cv2.imwrite(str(num)+".jpg",res)


    j = 0
    x_values = []
    y_values = []
    for i in shortedContours:
        x_values.append(j)
        y_values.append(cv2.contourArea(i))
        j = j + 1
    #plt.cla()
    #plt.clf()
    #plt.plot(x_values, y_values, "o")
    #plt.draw()


    #print (text)
    if(len(filteredContours) != num):
        return size

#seged fuggveny a forgatashoz
def Angle(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))
#seged fuggveny a trackerhez
def nothing(x):
    pass


def setAndGetPos():

    #trackbar rajzolasa
    cv2.namedWindow('dataSet')
    cv2.createTrackbar('threshParam1', 'dataSet', 150, 255, nothing)
    cv2.createTrackbar('threshParam2', 'dataSet', 250, 255, nothing)
    cv2.createTrackbar('cannyParam1', 'dataSet', 150, 255, nothing)
    cv2.createTrackbar('cannyParam2', 'dataSet', 200, 255, nothing)


    x = 0
    y = 0
    w = 0
    h = 0
    bool=False
    while True:
        #trackbar adatok lekerese
        threshParam1 = cv2.getTrackbarPos('threshParam1', 'dataSet')
        threshParam2 = cv2.getTrackbarPos('threshParam2', 'dataSet')
        cannyParam1 = cv2.getTrackbarPos('cannyParam1', 'dataSet')
        cannyParam2 = cv2.getTrackbarPos('cannyParam2', 'dataSet')

        cropped=""
        #uj kep
        ret, LiveImg = cap.read()
        #elforgatas szoge
        rot=0
        #marker kereses
        try:
            markers = detect_markers(LiveImg)
        except:
            continue
        #ha van marker es 3663 akkor
        for marker in markers:
            if (marker.id == 3663):
                #szog es pozicio eltarolasa
                rot = Angle(marker.contours[0][0], marker.contours[1][0])
                if(rot>45):
                    rot=rot-90

                x, y, w, h = cv2.boundingRect(marker.contours)  # good
                #kep vagasa az elonezeti kephez
                cropped = LiveImg[y:y + h, x:x + w]  # good
                offset = 50
                #pozico ellenorzes

                if (marker.contours[0][0][0] - offset < marker.contours[1][0][0] and marker.contours[0][0][0] + offset >
                    marker.contours[1][0][0] and marker.contours[1][0][1] - offset < marker.contours[2][0][1] and
                                marker.contours[1][0][1] + offset > marker.contours[2][0][1]):

                    bool=True


        #data rajzolsa, ezt majd megoldom mashogy
        LiveImg = cv2.resize(LiveImg, (640, 480), interpolation=cv2.INTER_AREA)
        ret, thresh1 = cv2.threshold(LiveImg, threshParam1,threshParam2, cv2.THRESH_BINARY)
        edge = cv2.Canny(thresh1, cannyParam1, cannyParam2)
        edge = cv2.resize(edge, (640, 480), interpolation=cv2.INTER_AREA)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        try:
            cropped = cv2.resize(cropped, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow("data", np.vstack((np.hstack((LiveImg,thresh1)),np.hstack((edge, cropped)))))
        except :
            tmp = np.zeros((height, width, 3), np.uint8)
            tmp = cv2.resize(tmp, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow("data", np.vstack((np.hstack((LiveImg, thresh1)), np.hstack((edge, tmp)))))


        if (cv2.waitKey(1) & 0xFF == ord('q') or bool==True):
            #eltolas
            o=25
            x=x+o
            y=y+o
            w=w-o*2
            h=h-o*2

            return x, y, w, h,rot,threshParam1,threshParam2,cannyParam1,cannyParam2



if __name__ == '__main__':
     plt.show()

#     screen = screeninfo.get_monitors()[screen_id]
     #width, height = screen.width, screen.height

     numOfSplit = 3
     res = []

     #kepernyo felbontas
     width= 1680
     height = 1050

     #a terulet detektalasahoz hasznalt marker betoltese es atmeretezese, kesobb meg a /2 helyere rakok egy valtozot
     maker = cv2.resize(cv2.imread("m.png"), (width/numOfSplit, height/numOfSplit))

     #marker ablak letherohaza
     cv2.resizeWindow('maker', width/numOfSplit, height/numOfSplit)
     cv2.imshow("maker",maker)
     cv2.moveWindow('maker', -1920, 0)

     #data ablak letherohaza
     cv2.namedWindow("data") 
     cv2.moveWindow('data', 0, 0)

     #video forras
     source= "http://10.224.83.34:8080/video"
     source = 1
     cap = cv2.VideoCapture(source)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, -1)

     cap.set(3, 1920)
     cap.set(4, 1080)
     """
     cv2.namedWindow('FocusSet')
     cv2.createTrackbar('FOCUS', 'FocusSet', 0, 500, nothing)
     while (True):
         # Capture frame-by-frame
         FOCUS = cv2.getTrackbarPos('FOCUS', 'dataSet') / 10
         cap.set(cv2.CAP_PROP_FOCUS, FOCUS)
         cv2.imshow('frame', cap.read()[1])
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break
     """

     c = False

     #ablak poziciojahoz hasznalt segedvaltozo
     statei=0
     statej=0

     score = []
     #main loop
     while True:
         #marker rajzolasa es pozicionalasa
         cv2.imshow("maker", maker)



         w=width/ numOfSplit
         h=height/numOfSplit

         if statei==numOfSplit:
             statei=1
             statej=statej+1
         else:
             statei=statei+1

         if(statei >numOfSplit or statej>=numOfSplit):
             print (score)
             exit()

         print ("New Pos: ",-statei*w, statej*h)
         cv2.moveWindow('maker', -statei*w, statej*h)

         #indulo pixelmeret
         size = 20
         multiplier = 3
         #marker poziciojanak, szogenek lekerdezese
         #tresh,canny param beallitasa

         x, y, w, h, rot, threshParam1, threshParam2, cannyParam1, cannyParam2 = setAndGetPos()

         print ("rot ",rot)
         #adott frame feldolgozasa
         while (True):
            #pixel meret csokkentes
            size = size-1

            #vonalak generalasa,rajzolasa
            lines,num = generateRectangle(size, height/numOfSplit,width/numOfSplit,rot)
            cv2.imshow("maker",lines)

            import time
            cv2.waitKey(200)

            #azert van ennyire szukseg hogy a video buffer biztosan kiuruljon
            cap.read()
            cap.read()
            cap.read()

            ret, LiveImg = cap.read()
            #vagas
            cropped = LiveImg[y:y + h, x:x + w]

            #kirajzolt vonalak detektalasa
            res = detv2(cropped, num,threshParam1,threshParam2,cannyParam1,cannyParam2)

            #ha elbukott az adott pixelszelesseggel akkor kov iteracio kezdete
            if res:
                score.append(res)
                cv2.waitKey(200)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                capture.release()
                cv2.destroyAllWindows()
    #ha meg volt mind a negy ablak pos akkor kilep
     if state ==3:
        exit()