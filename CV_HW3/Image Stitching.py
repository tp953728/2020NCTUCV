import cv2
import math
import numpy as np

img_list = [('1.jpg','2.jpg',0),('hill1.jpg','hill2.jpg',0),('S1.jpg','S2.jpg',0.3),('annapolis2.jpg','annapolis3.jpg',0),('wheel1.jpg','wheel2.jpg',0),('palace1.jpg','palace2.jpg',0),('annapolis1.jpg','annapolis2_annapolis3.jpg',0)]

def sift(input_img, img_name):
    img = np.uint8(input_img)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, des) = descriptor.detectAndCompute(img, None)
    cv2.drawKeypoints(img, kps, img, (0,255,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('./results/sift/sift_{}.jpg'.format(img_name.split('.')[0]),img)
    return kps, des

def find_match(des1, des2):
    dist_matrix = np.zeros((len(des1), len(des2)))
    # find the distance between each and every descriptors
    for index1 in range(len(des1)):
        for index2 in range(len(des2)):
            dist = np.linalg.norm(des1[index1] - des2[index2])
            dist_matrix[index1][index2] = dist

    #sort the vector to get the two best match with the smallest distance
    #match = [descriptor's number(descriptor i), the two best match with descriptor i, distance with the first match, destance with the second match]
    match = []
    for i in range(dist_matrix.shape[0]):
        two_best_match = np.argsort(dist_matrix[i])[:2]
        match.append ([i, two_best_match, dist_matrix[i][two_best_match[0]], dist_matrix[i][two_best_match[1]]])

    #perform ratio distance with ratio = 0.75
    final_match = []
    for i in range(len(match)):
        if match[i][2] < match[i][3] * 0.75:
            final_match.append([match[i][0], match[i][1][0]])
    return final_match

def draw_match(img1, img2, kps1, kps2, match, img_name):
    height1 = img1.shape[0]
    width1 = img1.shape[1]
    height2 = img2.shape[0]
    width2 = img2.shape[1]

    output_img = np.zeros((max(height1,height2), width1+width2, 3), dtype = 'uint8')
    output_img[0:height1, 0:width1] = img1
    output_img[0:height2, width1:] = img2

    for index1, index2 in match:
        color = list(map(int, np.random.randint(0, high=255, size=(3,))))
        pts1 = (int(kps1[index1].pt[0]), int(kps1[index1].pt[1]))
        pts2 = (int(kps2[index2].pt[0] + width1), int(kps2[index2].pt[1]))
        cv2.line(output_img, pts1, pts2, color, 1)

    cv2.imwrite('./results/match/matchline_'+img_name+'.jpg', output_img)

def homomat(min_match_count: int, src, dst):
    A = np.zeros((min_match_count * 2, 9))
    # construct the two sets of points
    for i in range(min_match_count):
        src1, src2 = src[i, 0, 0], src[i, 0, 1]
        dst1, dst2 = dst[i, 0, 0], dst[i, 0, 1]
        A[i * 2, :] = [src1, src2, 1, 0, 0, 0, -src1 * dst1, - src2 * dst1, -dst1]
        A[i * 2 + 1, :] = [0, 0, 0, src1, src2, 1, -src1 * dst2, - src2 * dst2, -dst2]
    
    # compute the homography between the two sets of points
    [_, S, V] = np.linalg.svd(A)
    m = V[np.argmin(S)]
    m *= 1 / m[-1]
    H = m.reshape((3, 3))
    return H

def ransac(final_match, kps_list, min_match_count, num_test: int, threshold: float):
    if len(final_match) > min_match_count:
        src_pts = np.array([kps_list[1][m[1]].pt for m in final_match]).reshape(-1, 1, 2)
        dst_pts = np.array([kps_list[0][m[0]].pt for m in final_match]).reshape(-1, 1, 2)
        min_outliers_count = math.inf
        
        while(num_test != 0):
            indexs = np.random.choice(len(final_match), min_match_count, replace=False)
            homography = homomat(min_match_count, src_pts[indexs], dst_pts[indexs])

            # Warp all left points with computed homography matrix and compare SSDs
            src_pts_reshape = src_pts.reshape(-1, 2)
            one = np.ones((len(src_pts_reshape), 1))
            src_pts_reshape = np.concatenate((src_pts_reshape, one), axis=1)
            warped_left = np.array(np.mat(homography) * np.mat(src_pts_reshape).T)
            for i, value in enumerate(warped_left.T):
                warped_left[:, i] = (value * (1 / value[2])).T

            # Calculate SSD
            dst_pts_reshape = dst_pts.reshape(-1, 2)
            dst_pts_reshape = np.concatenate((dst_pts_reshape, one), axis=1)
            inlier_count = 0
            inlier_list = []
            for i, pair in enumerate(final_match):
                ssd = np.linalg.norm(np.array(warped_left[:, i]).ravel() - dst_pts_reshape[i])
                if ssd <= threshold:
                    inlier_count += 1
                    inlier_list.append(pair)

            if (len(final_match) - inlier_count) < min_outliers_count:
                min_outliers_count = (len(final_match) - inlier_count)
                best_homomat = homography
                best_matches = inlier_list
            num_test -= 1
        return best_homomat, best_matches
    else:
        raise Exception("Not much matching keypoints exits!")

def blend(res, name, alpha=0.5):
    res = res.copy()
    for m in range(img1.shape[1]):
        for n in range(img1.shape[0]):
            if sum(res[n+top, m+left]) != 0:
                res[n+top, m+left] = alpha * res[n+top, m+left] + (1 - alpha) * img1[n, m]
            else:
                res[n+top, m+left] = img1[n, m]
    cv2.imwrite('./results/warp/warp_'+name+img_name+'.jpg', res)

def linear_blend(res, window_size=0):
    res = res.copy()
    alpha = step_a = window = step_w = 0
    for m in range(img1.shape[1]):
        alpha += step_a
        window += step_w
        for n in range(img1.shape[0]):
            if sum(res[n+top, m+left]) != 0:
                if window==0:
                    step_w = 1/(img1.shape[1]-m)
                    window+=step_w
                if window > window_size and alpha==0:
                    step_a = 1/(img1.shape[1]-m)
                    alpha+=step_a
                res[n+top, m+left] = alpha * res[n+top, m+left] + (1 - alpha) * img1[n, m]
            else:
                res[n+top, m+left] = img1[n, m]
    cv2.imwrite('./results/warp/linear_window_warp_'+img_name+'.jpg', res)
    return res

def forward_warp(h, w, warped_img, img_grid):
    res = np.zeros((h, w, 3), dtype='uint8')
    for x, y, im in zip(warped_img[0], warped_img[1], img_grid):
        res[math.floor(y)+top, math.floor(x)+left] = img2[im[1], im[0], :]
    cv2.imwrite('./results/warp/warp_res/forward_'+img_name+'.jpg', res)
    blend(res, 'forward_')

def inverse_warp(h, w, corners, homography):
    
    res_nn = np.zeros((h, w, 3), dtype='uint8')
    res_bi = np.zeros((h, w, 3), dtype='uint8')

    # Create image 2 grid in image 1 coordinate
    b, t, r, l = math.ceil(max(corners[:, 1])),math.floor(min(corners[:, 1])),math.ceil(max(corners[:, 0])), math.floor(min(corners[:, 0]))

    img2_trans_grid = [[n, m, 1] for n in range(l, r) for m in range(min(t, 1), b)]
    
    # Inverse mapping points on image 1 to image 2
    img2_trans_inv = np.array(np.mat(np.linalg.inv(homography)) * np.mat(img2_trans_grid).T)
    img2_trans_inv /= img2_trans_inv[2]

    for x, y, im in zip(img2_trans_inv[0], img2_trans_inv[1], img2_trans_grid):
        if math.ceil(y) < img2.shape[0] and math.ceil(y)>0 and math.ceil(x) < img2.shape[1] and math.ceil(x)>0:
            # Nearest Neighbor
            res_nn[im[1]+top, im[0]+left] = img2[int(y + 0.5), int(x + 0.5), :]
            
            # Bilinear interpolation
            res_bi[im[1]+top,im[0]+left] = (img2[math.ceil(y), math.ceil(x), :]*((y-math.floor(y))*(x-math.floor(x)))+
                                                 img2[math.floor(y), math.floor(x), :]*((math.ceil(y)-y)*(math.ceil(x)-x))+
                                                 img2[math.ceil(y), math.floor(x), :]*((y-math.floor(y))*(math.ceil(x)-x))+
                                                 img2[math.floor(y), math.ceil(x), :]*((math.ceil(y)-y)*(x-math.floor(x))))
    cv2.imwrite('./results/warp/warp_res/backward_nn_'+img_name+'.jpg', res_nn)
    cv2.imwrite('./results/warp/warp_res/backward_bi_'+img_name+'.jpg', res_bi)
    #blend(res_nn, 'inverse_nn_', 0.5)
    #blend(res_bi, 'inverse_bi_', 0.5)
    return linear_blend(res_bi, window_size)

def warp(homography):
    
    # Transform image 2 with homography matrix
    img2_grid = [[n, m, 1] for n in range(img2.shape[1]) for m in range(img2.shape[0])]
    img2_trans = np.array(np.mat(homography) * np.mat(img2_grid).T)
    img2_trans /= img2_trans[2]

    # Find transformed four corners of image 2 on image 1 coordinate system

    corners = np.zeros((4, 3))
    for p, im in zip(img2_trans.T, img2_grid):
        if im[0] == 0 and im[1] == 0:
            corners[0] = p
        elif im[0] == 0 and im[1] == img2.shape[0] - 1:
            corners[1] = p
        elif im[0] == img2.shape[1] - 1 and im[1] == 0:
            corners[2] = p
        elif im[0] == img2.shape[1] - 1 and im[1] == img2.shape[0] - 1:
            corners[3] = p
            
    # Blended image size
    global top,left,bottom,r
    top = max(0, math.ceil(-min(corners.T[1])))
    bottom = max(img1.shape[0], math.ceil(max(corners.T[1])))+top
    left = max(0, math.ceil(-min(corners.T[0])))
    right = max(img1.shape[1], math.ceil(max(corners.T[0])))+left
    r = max(img1.shape[1], math.ceil(min(corners[2][0],corners[3][0])))+left
        
    # Forward warping
    #forward_warp(bottom, right, img2_trans, img2_grid)
    
    # Inverse warping
    return inverse_warp(bottom, right, corners, homography)

for img in img_list:
    img1 = cv2.imread('./data/{}'.format(img[0]))
    img2 = cv2.imread('./data/{}'.format(img[1]))
    
    top = left = bottom = r = 0
    window_size=img[2]
    
    if img[0]=='hill1.jpg':
        img1 = img1[1:,1:,:]
        img2 = img2[1:,1:,:]
    
    # kps = keypoints, des = descriptor
    kps1, des1 = sift(cv2.imread('./data/{}'.format(img[0])), img[0])
    kps2, des2 = sift(cv2.imread('./data/{}'.format(img[1])), img[1])

    # feature matching
    match = find_match(des1, des2)

    # draw the matching result (by drawing lines between mathing points) to check the result
    img_name = img[0].split('.')[0] + '_' + img[1].split('.')[0]
    draw_match(img1, img2, kps1, kps2, match, img_name)
    
    # RANSAC
    best_homo, best_match = ransac(match, [kps1,kps2], 8, 1000, 0.5)
    draw_match(img1, img2, kps1, kps2, best_match, img_name+'_best_match')
    
    # WRAP
    res_img = warp(best_homo)
    
    # CROP
    while([0,0,0] in res_img[top,left:r,:]):
        top+=1
    while([0,0,0] in res_img[bottom-1,left:r,:]):
        bottom-=1
    cv2.imwrite('./results/final/'+img_name+'.jpg', res_img[top:bottom,left:r,:])