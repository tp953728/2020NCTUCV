import cv2
import math
import scipy.io
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
eng = matlab.engine.start_matlab()

img_list = [['Mesona1.JPG','Mesona2.JPG'],['Statue1.bmp','Statue2.bmp'],['box1.JPG','box2.JPG'],['pvc1.JPG','pvc2.JPG']]

def sift(input_img, img_name):
    img = np.uint8(input_img)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, des) = descriptor.detectAndCompute(img, None)
    for kp in kps:
        color = tuple(np.random.randint(0,255,3).tolist())
        point = tuple([int(p) for p in kp.pt])
        cv2.circle(img, point, 5, color, -1)
    cv2.imwrite('./results/sift/{}.jpg'.format(img_name.split('.')[0]),img)
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

def normalize(p):
    mat = np.matrix(p)
    mean = np.array(mat.mean(0)).flatten()
    std = np.array(mat.std(0)).flatten()
    T = np.matrix([[1/std[0], 0      , -mean[0]/std[0]],
                   [0       ,1/std[1], -mean[1]/std[1]],
                   [0       , 0      , 1]])
    return np.array((T*mat.T).T), T

def fundamental_matrix(p1, p2, threshold=0.05):
    # Normalize
    p1n, T1 = normalize(p1)
    p2n, T2 = normalize(p2)
    
    # RANSAC
    niter = 3000
    maxinliner = 0
    while(niter>0):
        # 8-point algorithm
        idx = np.random.randint(0,p1n.shape[0],8)
        pts1 = p1n[idx]
        pts2 = p2n[idx]

        # get the system of equations
        A = np.zeros((8,9))
        for i in range(8):
            x1, y1,_ = pts1[i]
            x2, y2,_ = pts2[i]
            A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

        # solve f from Af=0 using SVD
        U, S, V = np.linalg.svd(A)
        F = V[-1].reshape((3, 3))

        # Resolve det(F)=0 contraint using SVD
        U, S, V = np.linalg.svd(F)
        S[-1] = 0
        F = U.dot(np.diag(S).dot(V))

        inliner_list = []
        for i in range(p1n.shape[0]):
            error = p2n[i, :].dot(F.dot(p1n[i, :]).T)
            if abs(error) <= threshold:
                inliner_list.append([p1n[i, :], p2n[i, :]])

        if len(inliner_list) > maxinliner:
            maxinliner = len(inliner_list)
            best_F = F
            best_inliner = inliner_list
        niter -= 1
    
    # De-normalize
    H = T2.T.dot(best_F.dot(T1))
    H /= H[2,2]
    H_inliner = []
    for pair in best_inliner:
        H_1 = np.array(np.linalg.inv(T1) * np.mat(pair[0]).T).flatten().T
        H_1 /= H_1[2]
        H_2 = np.array(np.linalg.inv(T2) * np.mat(pair[1]).T).flatten().T
        H_2 /= H_2[2]
        H_inliner.append([H_1, H_2])
    return np.array(H), np.array(H_inliner)

def compute_epilines(H, pts):
    res = np.ones((pts.shape[0], 3))
    for idx, point in enumerate(pts):
        temp = H.dot(point)
        temp *= 1/np.sqrt(temp[0]**2+temp[1]**2)
        res[idx] = temp
    return res

def essential_matrix(K1,K2,H):
    E = np.dot(np.dot(K1.T,H),K2)
    U,S,V = np.linalg.svd(E)
    m = (S[0]+S[1])/2
    E = np.dot(np.dot(U, np.diag((m,m,0))), V)
    return E

def get_4_possible_projection_matrix(E):
    U, _, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
           np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s

def get_correct_P(x1, x2, P1, P2):
    C = np.dot(P2[:,0:3], P2[:,3].T)
    tripoints3d = triangulation(x1, x2, P1, P2)
    infront = 0
    for i in range(tripoints3d.shape[1]):
        if np.dot((i - C), P2[:,2].T) > 0:
            infront += 1
    return infront

def triangulation(x1, x2, P1, P2):
    res = np.ones((x1.shape[1], 4))
    for i in range(x1.shape[1]):
        A = np.asarray([
            (x1[0, i] * P1[2, :].T - P1[0, :].T),
            (x1[1, i] * P1[2, :].T - P1[1, :].T),
            (x2[0, i] * P2[2, :].T - P2[0, :].T),
            (x2[1, i] * P2[2, :].T - P2[1, :].T)
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1]
        res[i, :] = X / X[3]
    return res.T

def draw(img1, img2, lines, pts1, pts2):
    r, c = img2.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.circle(img1, tuple(pt1[:2]), 5, color, -1)
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        img2 = cv2.circle(img2, tuple(pt2[:2]), 5, color, -1)
    img = np.concatenate((img1,img2),axis=1)
    cv2.imwrite('./results/epiline/{}.jpg'.format(img_name),img)

k = np.array([[1.4219, 0.0005, 0.5092],
              [0.0000, 1.4219, 0.3802],
              [0.0000, 0.0000, 0.0010]])

k1 = np.array([[5426.566895, 0.678017, 330.096680],
               [0.000000, 5423.133301, 648.950012],
               [0.000000, 0.000000, 1.000000]])

k2 = np.array([[5426.566895, 0.678017, 387.430023],
               [0.000000, 5423.133301, 620.616699],
               [0.000000, 0.000000, 1.000000]])

k3 = np.array([[17586.8128, 0.00000000, 1528.83403],
               [0.00000000, 17647.3547, 2076.61701],
               [0.00000000, 0.00000000, 1.00000000]])

k4 = np.array([[4.83426616, 0.00000000, 1.39992570],
               [0.00000000, 4.90741870, 2.09099732],
               [0.00000000, 0.00000000, 0.00100000]])
k1*=0.001
k2*=0.001
k3*=0.001

for img in img_list:
    img1 = cv2.imread('./data/{}'.format(img[0]))
    img2 = cv2.imread('./data/{}'.format(img[1]))
    img_name = img[0].split('.')[0]
    threshold = 0.05

    if img_name == 'Mesona1':
        K1 = K2 = k
        threshold = 0.005
    elif img_name == 'Statue1':
        K1 = k1
        K2 = k2
        threshold = 0.001
    elif img_name == 'box1':
        K1 = K2 = k3
        threshold = 0.005
    elif img_name == 'pvc1':
	K1 = K2 = k4
	threshold = 0.001

    # kps = keypoints, des = descriptor
    kps1, des1 = sift(cv2.imread('./data/{}'.format(img[0])), img[0])
    kps2, des2 = sift(cv2.imread('./data/{}'.format(img[1])), img[1])

    # feature matching
    match = find_match(des1, des2)

    # Find out correspondence across imgs
    correspondence_img1 = [kps1[x[0]].pt for x in match]
    correspondence_img2 = [kps2[x[1]].pt for x in match]

    correspondence_img1 = np.concatenate((correspondence_img1, np.ones((len(correspondence_img1), 1))), axis=1)
    correspondence_img2 = np.concatenate((correspondence_img2, np.ones((len(correspondence_img2), 1))), axis=1)

    # Estimate the fundamental matrix
    H, H_inliner = fundamental_matrix(correspondence_img1, correspondence_img2, threshold)
    pts1 = H_inliner[:,0]
    pts2 = H_inliner[:,1]


    #H, mask = cv2.findFundamentalMat(correspondence_img1, correspondence_img2, cv2.FM_LMEDS)
    #pts1 = correspondence_img1[mask.ravel() == 1]
    #pts2 = correspondence_img2[mask.ravel() == 1]


    # Compute the corresponding epipolar lines
    epilines = compute_epilines(H, pts1)

    # Draw interesting pts and epipolar lines
    img1_grey = cv2.imread('./data/{}'.format(img[0]), cv2.IMREAD_GRAYSCALE)
    img2_grey = cv2.imread('./data/{}'.format(img[1]), cv2.IMREAD_GRAYSCALE)
    draw(img1_grey, img2_grey, epilines, pts1.astype('int'), pts2.astype('int'))

    # Get Essential Matrix
    E = essential_matrix(K1,K2,H)

    # Get 4 possible camera paramters
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    camera_matrix_1 = np.dot(K1,P1)
    P2s = get_4_possible_projection_matrix(E)

    # Get the right projection matrix
    maxinfornt = 0
    for i, P2 in enumerate(P2s):
        P2 = np.dot(K2,P2)
        infront = get_correct_P(pts1.T, pts2.T, camera_matrix_1, P2)
        if infront > maxinfornt:
            maxinfornt = infront
            ind = i
            camera_matrix_2 = P2
    print("best projection matrix index: ", ind)

    # Apply triangulation to get 3D points
    tripoints3d = triangulation(pts1.T, pts2.T, camera_matrix_1, camera_matrix_2)

    # Show world points
    #get_ipython().run_line_magic('matplotlib', 'notebook')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tripoints3d[0], tripoints3d[1], tripoints3d[2], c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=135, azim=90)
    plt.show()

    # Texture mapping to get 3D model
    eng.obj_main(matlab.double(tripoints3d.T[:,:3].tolist()), matlab.double(pts1.tolist()), matlab.double(camera_matrix_1.tolist()), './data/{}'.format(img[0]), 1.0, nargout=0)



