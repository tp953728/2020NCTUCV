import os
import sys
import time, datetime
import numpy as np
#from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
from PIL import Image
import matplotlib.pyplot as plt

# calculate Mean Square Error
# the Mean Square Error between the two images is 
# the sum of the square difference between the two images
# NOTE: the two images must have the same dimension
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    #print imageA.astype("float")
    # return the MSE
    # the lower the error, the more similar the two images are    
    return err

# find how much image shifts, and return the shifted value   
def findShift(basicImg, matchingImg):
    basicN, basicM = basicImg.shape   # N:height M:width
    matchN, matchM = matchingImg.shape # N:height M:width
    
    #print "basic:", basicN, basicM
    #print "match:", matchN, matchM
    
    xShifted = 0
    yShifted = 0
    err = mse(basicImg[:matchN,:matchM], matchingImg)
    
    # find the smallest err and how much basicImg need to shift
    # such that the two images could best match 
    for i in range(basicN-matchN):
        for j in range(basicM-matchM):
            temp = mse(basicImg[i:matchN+i, j:matchM+j], matchingImg)
            #print temp
            
            if(temp < err):
                err = temp
                xShifted = i
                yShifted = j
                
    return xShifted, yShifted, err


def findShiftWithShifted(basicImg, matchingImg, xShifted, yShifted, err):
    matchN, matchM = matchingImg.shape
    xShiftedReturn = xShifted
    yShiftedReturn = yShifted
    err = mse(basicImg[xShifted: xShifted + matchN, yShifted : yShifted + matchM], matchingImg)
    
    searchRangeX = min(xShifted, 2)
    searchRangeY = min(yShifted, 2)
    #print "searchRange:", searchRangeX, searchRangeY
    
    for i in range(-searchRangeX, searchRangeX):
        for j in range(-searchRangeY, searchRangeY):
            temp = mse(basicImg[xShifted + i : xShifted + matchN + i, yShifted + j : yShifted + matchM + j], matchingImg)
            
            if(temp < err):
                err = temp
                xShiftedReturn = xShifted + i
                yShiftedReturn = yShifted + j
                
    return xShiftedReturn, yShiftedReturn, err

# do image pyramid and down sampling several layers of small images
# match the smallest image and find the shift
# up sampling the image and adjust the shift until the original image
def downSearch(basicImg, matchingImg, sub_rate, iteration):
    xShifted = 0
    yShifted = 0
    basicPyramid = []
    matchingPyramid = []
    
    basicPyramid.append(np.array(basicImg))
    matchingPyramid.append(np.array(matchingImg))
    
    #basicImg = Image.fromarray(basicImgArray).convert("L")
    #matchingImg = Image.fromarray(matchingImgArray).convert("L")
    for i in range(iteration):
        n, m = basicImg.size
        basicImg = basicImg.resize((int(n/sub_rate), int(m/sub_rate)))
        basicPyramid.append(np.array(basicImg))
        
        n, m = matchingImg.size
        matchingImg = matchingImg.resize((int(n/sub_rate), int(m/sub_rate)))
        matchingPyramid.append(np.array(matchingImg))
        
    xShifted, yShifted, err = findShift(basicPyramid[iteration], matchingPyramid[iteration])
    #print "after findShift: ", xShifted, yShifted, err
    
    for i in range(iteration-1, -1, -1):
        xShifted *= sub_rate
        yShifted *= sub_rate
        xShifted, yShifted, err = findShiftWithShifted(basicPyramid[i], matchingPyramid[i], xShifted, yShifted, err)
        #xShifted, yShifted, err = findShift(basicPyramid[i], matchingPyramid[i])
        #print "iteration:", i, ":", xShifted, yShifted, err
    
    return xShifted, yShifted, err
    
if __name__ == "__main__":
    time_start=time.time()
    print("Aligning the G and R channels to the B channel of " + "\"" + sys.argv[1] + "\"" + "...")
    img = Image.open(sys.argv[1])
    oriWidth, oriHeight = img.size
    #print oriWidth
    maxHeight = int(oriHeight/3) + 1 if oriHeight % 3 == 1 else int(oriHeight/3)
    #print img.mode
    
    # crop(left, upper, right, lower)
    # devide image into the three b g r channels
    b = img.crop((0,                     0, oriWidth,             maxHeight))
    g = img.crop((0,             maxHeight, oriWidth, oriHeight - maxHeight))
    r = img.crop((0, oriHeight - maxHeight, oriWidth,             oriHeight))
    
    #print "type:", r.mode
    #plt.imshow(r)
    #plt.show()
    #plt.imshow(g)
    #plt.show()
    #plt.imshow(b)
    #plt.show()
    #print g.size

    cut = int(oriWidth / 21.0) # cut the boarder
    #print "cut:", cut
    
    sub_rate = 2
    iteration = 0
    while cut / (2 ** iteration) > 10:
        iteration += 1
    print "iteration:", iteration
    
    imgWidth, imgHeight = g.size
    gCut = g.crop((cut, cut, imgWidth - cut, imgHeight - cut))
    gxShifted, gyShifted, gErr = downSearch(r, gCut, sub_rate, iteration)
    #print gxShifted, gyShifted, gErr
   
    
    imgWidth, imgHeight = b.size
    bCut = b.crop((cut, cut, imgWidth - cut, imgHeight - cut))
    bxShifted, byShifted, bErr = downSearch(r, bCut, sub_rate, iteration)
    #print bxShifted, byShifted, bErr
    
    rxShifted = 0
    ryShifted = 0
    gxShifted -= cut
    gyShifted -= cut
    bxShifted -= cut
    byShifted -= cut

    if min(gxShifted, bxShifted) < 0:
        temp = min(gxShifted, bxShifted)
        rxShifted -= temp
        gxShifted -= temp
        bxShifted -= temp
    if min(gyShifted, byShifted) < 0:
        temp = min(gyShifted, byShifted)
        ryShifted -= temp
        gyShifted -= temp
        byShifted -= temp
    xExtend = max(rxShifted, gxShifted, bxShifted)
    yExtend = max(ryShifted, gyShifted, byShifted)
    
    # create a new image with the given mode and size
    # L mean gray img that each pixel has 8 bits
    # 0 for black 255 for white
    # L = R * 299/1000 + G * 587/1000+ B * 114/1000
    newRimg = Image.new('L', (oriWidth + xExtend, maxHeight + yExtend))
    newGimg = Image.new('L', (oriWidth + xExtend, maxHeight + yExtend))
    newBimg = Image.new('L', (oriWidth + xExtend, maxHeight + yExtend))
    
    #print "R shift:", rxShifted, ryShifted
    #print "G shift:", gxShifted, gyShifted
    #print "B shift:", bxShifted, byShifted
    
    # Split this image into individual bands. 
    # This method returns a tuple of individual image bands from an image. 
    # For example, splitting an RGB image creates three new images 
    # each containing a copy of one of the original bands (red, green, blue).

    # L mean gray img that each pixel has 32 bits
    # 0 for black 255 for white
    # I = R * 299/1000 + G * 587/1000 + B * 114/1000
    if img.format == "TIFF": # check the format of input dataset
        _, depth = img.mode.split(';') # seperated by ';' (I;depth)
        depth = float(depth) # 16
        # normalize 16 bits and extend it into 8 bits
        newRimg.paste(Image.fromarray(np.array(r) / (2 ** depth - 1) * 255.0).convert("L"), (ryShifted, rxShifted))
        #print "r: ",r
        newGimg.paste(Image.fromarray(np.array(g) / (2 ** depth - 1) * 255.0).convert("L"), (gyShifted, gxShifted))
        #print "g: ",g
        newBimg.paste(Image.fromarray(np.array(b) / (2 ** depth - 1) * 255.0).convert("L"), (byShifted, bxShifted))
        #print "b: ",b
    else:
        # for other img format
        newRimg.paste(r, (ryShifted, rxShifted))
        newGimg.paste(g, (gyShifted, gxShifted))
        newBimg.paste(b, (byShifted, bxShifted))
    
    # calculate the execution time 
    time_end=time.time()
    print "execution time: ",time_end-time_start, "s"

    final = Image.merge("RGB", [newRimg, newGimg, newBimg])
    result_path = os.path.abspath('.') + "/colorizing"  
    final.save(os.path.join(result_path, sys.argv[1] + "_result.bmp"))
    plt.title('Result')
    plt.imshow(final)
    plt.show()


