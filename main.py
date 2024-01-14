import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import null_space

def compute_coefficients(data):
    coeffs = np.zeros((12))
    coeffs[0] = data[4]*data[6]*data[29]*data[35]*data[47]**2 - data[3]*data[7]*data[29]*data[35]*data[47]**2 - data[1]*data[6]*data[30]*data[35]*data[47]**2 + data[0]*data[7]*data[30]*data[35]*data[47]**2 + data[6]**2*data[27]**2*data[36] + 2*data[6]*data[7]*data[27]*data[28]*data[36] + data[7]**2*data[28]**2*data[36] + data[1]*data[3]*data[35]*data[47]**2 - data[0]*data[4]*data[35]*data[47]**2 + 2*data[6]*data[8]*data[27]*data[36] + 2*data[7]*data[8]*data[28]*data[36] + data[8]**2*data[36]
    coeffs[1] = -data[7]*data[12]*data[29]*data[35]*data[47]**2 + data[6]*data[13]*data[29]*data[35]*data[47]**2 + data[4]*data[15]*data[29]*data[35]*data[47]**2 - data[3]*data[16]*data[29]*data[35]*data[47]**2 + data[7]*data[9]*data[30]*data[35]*data[47]**2 - data[6]*data[10]*data[30]*data[35]*data[47]**2 - data[1]*data[15]*data[30]*data[35]*data[47]**2 + data[0]*data[16]*data[30]*data[35]*data[47]**2 + 2*data[6]*data[15]*data[27]**2*data[36] + 2*data[7]*data[15]*data[27]*data[28]*data[36] + 2*data[6]*data[16]*data[27]*data[28]*data[36] + 2*data[7]*data[16]*data[28]**2*data[36] - data[4]*data[9]*data[35]*data[47]**2 + data[3]*data[10]*data[35]*data[47]**2 + data[1]*data[12]*data[35]*data[47]**2 - data[0]*data[13]*data[35]*data[47]**2 + 2*data[8]*data[15]*data[27]*data[36] + 2*data[6]*data[17]*data[27]*data[36] + 2*data[8]*data[16]*data[28]*data[36] + 2*data[7]*data[17]*data[28]*data[36] + 2*data[8]*data[17]*data[36]
    coeffs[2] = data[13]*data[15]*data[29]*data[35]*data[47]**2 - data[12]*data[16]*data[29]*data[35]*data[47]**2 - data[10]*data[15]*data[30]*data[35]*data[47]**2 + data[9]*data[16]*data[30]*data[35]*data[47]**2 + data[15]**2*data[27]**2*data[36] + 2*data[15]*data[16]*data[27]*data[28]*data[36] + data[16]**2*data[28]**2*data[36] + data[10]*data[12]*data[35]*data[47]**2 - data[9]*data[13]*data[35]*data[47]**2 + 2*data[15]*data[17]*data[27]*data[36] + 2*data[16]*data[17]*data[28]*data[36] + data[17]**2*data[36]
    coeffs[3] = -data[7]*data[21]*data[29]*data[35]*data[47]**2 + data[6]*data[22]*data[29]*data[35]*data[47]**2 + data[4]*data[24]*data[29]*data[35]*data[47]**2 - data[3]*data[25]*data[29]*data[35]*data[47]**2 + data[7]*data[18]*data[30]*data[35]*data[47]**2 - data[6]*data[19]*data[30]*data[35]*data[47]**2 - data[1]*data[24]*data[30]*data[35]*data[47]**2 + data[0]*data[25]*data[30]*data[35]*data[47]**2 + 2*data[6]*data[24]*data[27]**2*data[36] + 2*data[7]*data[24]*data[27]*data[28]*data[36] + 2*data[6]*data[25]*data[27]*data[28]*data[36] + 2*data[7]*data[25]*data[28]**2*data[36] - data[4]*data[18]*data[35]*data[47]**2 + data[3]*data[19]*data[35]*data[47]**2 + data[1]*data[21]*data[35]*data[47]**2 - data[0]*data[22]*data[35]*data[47]**2 + 2*data[8]*data[24]*data[27]*data[36] + 2*data[6]*data[26]*data[27]*data[36] + 2*data[8]*data[25]*data[28]*data[36] + 2*data[7]*data[26]*data[28]*data[36] + 2*data[8]*data[26]*data[36]
    coeffs[4] = -data[16]*data[21]*data[29]*data[35]*data[47]**2 + data[15]*data[22]*data[29]*data[35]*data[47]**2 + data[13]*data[24]*data[29]*data[35]*data[47]**2 - data[12]*data[25]*data[29]*data[35]*data[47]**2 + data[16]*data[18]*data[30]*data[35]*data[47]**2 - data[15]*data[19]*data[30]*data[35]*data[47]**2 - data[10]*data[24]*data[30]*data[35]*data[47]**2 + data[9]*data[25]*data[30]*data[35]*data[47]**2 + 2*data[15]*data[24]*data[27]**2*data[36] + 2*data[16]*data[24]*data[27]*data[28]*data[36] + 2*data[15]*data[25]*data[27]*data[28]*data[36] + 2*data[16]*data[25]*data[28]**2*data[36] - data[13]*data[18]*data[35]*data[47]**2 + data[12]*data[19]*data[35]*data[47]**2 + data[10]*data[21]*data[35]*data[47]**2 - data[9]*data[22]*data[35]*data[47]**2 + 2*data[17]*data[24]*data[27]*data[36] + 2*data[15]*data[26]*data[27]*data[36] + 2*data[17]*data[25]*data[28]*data[36] + 2*data[16]*data[26]*data[28]*data[36] + 2*data[17]*data[26]*data[36]
    coeffs[5] = data[22]*data[24]*data[29]*data[35]*data[47]**2 - data[21]*data[25]*data[29]*data[35]*data[47]**2 - data[19]*data[24]*data[30]*data[35]*data[47]**2 + data[18]*data[25]*data[30]*data[35]*data[47]**2 + data[24]**2*data[27]**2*data[36] + 2*data[24]*data[25]*data[27]*data[28]*data[36] + data[25]**2*data[28]**2*data[36] + data[19]*data[21]*data[35]*data[47]**2 - data[18]*data[22]*data[35]*data[47]**2 + 2*data[24]*data[26]*data[27]*data[36] + 2*data[25]*data[26]*data[28]*data[36] + data[26]**2*data[36]
    coeffs[6] = data[4]*data[6]*data[33]*data[41]*data[48]**2 - data[3]*data[7]*data[33]*data[41]*data[48]**2 - data[1]*data[6]*data[34]*data[41]*data[48]**2 + data[0]*data[7]*data[34]*data[41]*data[48]**2 + data[6]**2*data[31]**2*data[42] + 2*data[6]*data[7]*data[31]*data[32]*data[42] + data[7]**2*data[32]**2*data[42] + data[1]*data[3]*data[41]*data[48]**2 - data[0]*data[4]*data[41]*data[48]**2 + 2*data[6]*data[8]*data[31]*data[42] + 2*data[7]*data[8]*data[32]*data[42] + data[8]**2*data[42]
    coeffs[7] = -data[7]*data[12]*data[33]*data[41]*data[48]**2 + data[6]*data[13]*data[33]*data[41]*data[48]**2 + data[4]*data[15]*data[33]*data[41]*data[48]**2 - data[3]*data[16]*data[33]*data[41]*data[48]**2 + data[7]*data[9]*data[34]*data[41]*data[48]**2 - data[6]*data[10]*data[34]*data[41]*data[48]**2 - data[1]*data[15]*data[34]*data[41]*data[48]**2 + data[0]*data[16]*data[34]*data[41]*data[48]**2 + 2*data[6]*data[15]*data[31]**2*data[42] + 2*data[7]*data[15]*data[31]*data[32]*data[42] + 2*data[6]*data[16]*data[31]*data[32]*data[42] + 2*data[7]*data[16]*data[32]**2*data[42] - data[4]*data[9]*data[41]*data[48]**2 + data[3]*data[10]*data[41]*data[48]**2 + data[1]*data[12]*data[41]*data[48]**2 - data[0]*data[13]*data[41]*data[48]**2 + 2*data[8]*data[15]*data[31]*data[42] + 2*data[6]*data[17]*data[31]*data[42] + 2*data[8]*data[16]*data[32]*data[42] + 2*data[7]*data[17]*data[32]*data[42] + 2*data[8]*data[17]*data[42]
    coeffs[8] = data[13]*data[15]*data[33]*data[41]*data[48]**2 - data[12]*data[16]*data[33]*data[41]*data[48]**2 - data[10]*data[15]*data[34]*data[41]*data[48]**2 + data[9]*data[16]*data[34]*data[41]*data[48]**2 + data[15]**2*data[31]**2*data[42] + 2*data[15]*data[16]*data[31]*data[32]*data[42] + data[16]**2*data[32]**2*data[42] + data[10]*data[12]*data[41]*data[48]**2 - data[9]*data[13]*data[41]*data[48]**2 + 2*data[15]*data[17]*data[31]*data[42] + 2*data[16]*data[17]*data[32]*data[42] + data[17]**2*data[42]
    coeffs[9] = -data[7]*data[21]*data[33]*data[41]*data[48]**2 + data[6]*data[22]*data[33]*data[41]*data[48]**2 + data[4]*data[24]*data[33]*data[41]*data[48]**2 - data[3]*data[25]*data[33]*data[41]*data[48]**2 + data[7]*data[18]*data[34]*data[41]*data[48]**2 - data[6]*data[19]*data[34]*data[41]*data[48]**2 - data[1]*data[24]*data[34]*data[41]*data[48]**2 + data[0]*data[25]*data[34]*data[41]*data[48]**2 + 2*data[6]*data[24]*data[31]**2*data[42] + 2*data[7]*data[24]*data[31]*data[32]*data[42] + 2*data[6]*data[25]*data[31]*data[32]*data[42] + 2*data[7]*data[25]*data[32]**2*data[42] - data[4]*data[18]*data[41]*data[48]**2 + data[3]*data[19]*data[41]*data[48]**2 + data[1]*data[21]*data[41]*data[48]**2 - data[0]*data[22]*data[41]*data[48]**2 + 2*data[8]*data[24]*data[31]*data[42] + 2*data[6]*data[26]*data[31]*data[42] + 2*data[8]*data[25]*data[32]*data[42] + 2*data[7]*data[26]*data[32]*data[42] + 2*data[8]*data[26]*data[42]
    coeffs[10] = -data[16]*data[21]*data[33]*data[41]*data[48]**2 + data[15]*data[22]*data[33]*data[41]*data[48]**2 + data[13]*data[24]*data[33]*data[41]*data[48]**2 - data[12]*data[25]*data[33]*data[41]*data[48]**2 + data[16]*data[18]*data[34]*data[41]*data[48]**2 - data[15]*data[19]*data[34]*data[41]*data[48]**2 - data[10]*data[24]*data[34]*data[41]*data[48]**2 + data[9]*data[25]*data[34]*data[41]*data[48]**2 + 2*data[15]*data[24]*data[31]**2*data[42] + 2*data[16]*data[24]*data[31]*data[32]*data[42] + 2*data[15]*data[25]*data[31]*data[32]*data[42] + 2*data[16]*data[25]*data[32]**2*data[42] - data[13]*data[18]*data[41]*data[48]**2 + data[12]*data[19]*data[41]*data[48]**2 + data[10]*data[21]*data[41]*data[48]**2 - data[9]*data[22]*data[41]*data[48]**2 + 2*data[17]*data[24]*data[31]*data[42] + 2*data[15]*data[26]*data[31]*data[42] + 2*data[17]*data[25]*data[32]*data[42] + 2*data[16]*data[26]*data[32]*data[42] + 2*data[17]*data[26]*data[42]
    coeffs[11] = data[22]*data[24]*data[33]*data[41]*data[48]**2 - data[21]*data[25]*data[33]*data[41]*data[48]**2 - data[19]*data[24]*data[34]*data[41]*data[48]**2 + data[18]*data[25]*data[34]*data[41]*data[48]**2 + data[24]**2*data[31]**2*data[42] + 2*data[24]*data[25]*data[31]*data[32]*data[42] + data[25]**2*data[32]**2*data[42] + data[19]*data[21]*data[41]*data[48]**2 - data[18]*data[22]*data[41]*data[48]**2 + 2*data[24]*data[26]*data[31]*data[42] + 2*data[25]*data[26]*data[32]*data[42] + data[26]**2*data[42]


    coeffs_ind = [0,6,1,0,6,7,2,1,7,8,3,6,0,9,4,3,9,7,1,10,2,8,11,5,5,9,3,11,5,11,10,4,4,10,8,2]

    return coeffs, coeffs_ind

def tnull_space(A, rcond=None):

    u, s, vh = np.linalg.svd(A, full_matrices=False)
    M, N = A.shape 
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q



def setup_elimination_templates(data):
    
    coeffs, coeffs_ind = compute_coefficients(data)

    C_ind = [0, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 39, 40, 42, 45, 46, 47, 49, 50, 51, 52, 55, 56, 57, 58]
    C = np.zeros((6, 10))
    
    for i,c_i in enumerate(C_ind):
        col = math.ceil((c_i+1)/6) - 1
        row = c_i - col*6
        C[row,col] = coeffs[coeffs_ind[i]] 

    return C

def solverHomography2SIFT(data):
    C = setup_elimination_templates(data)
    C0 = C[:, :6]
    C1 = C[:, 6:]
    
    # Using np.linalg.solve for a more direct solution
    C1 = np.linalg.solve(C0, C1)

    RR = np.vstack([-C1[-2:], np.eye(4)])
    AM_ind = [4, 0, 5, 1]  # Adjust indices for Python (0-based)
    AM = RR[AM_ind, :]
    D, V = np.linalg.eig(AM)
    
    V = V /  V[0,:] * np.ones((V.shape[0],1))
    sols = np.empty((2,V.shape[1]), dtype=complex)
    sols[0,:] = V[1,:]
    sols[1,:] = D

    return sols



def normalizePoints(pts1, pts2):
    N = pts1.shape[0]

    mass_point1  =np.mean(pts1, axis=0)
    mass_point2 = np. mean(pts2, axis=0)
    
    normalized_src_points = pts1 - mass_point1
    normalized_dst_points = pts2 - mass_point2
    # print(normalized_src_points)
    avg_dist1 = np.mean(np.linalg.norm(normalized_src_points, axis=1))
    avg_dist2 = np.mean(np.linalg.norm(normalized_dst_points, axis=1))
    
    avg_ratio1 = np.sqrt(2)/avg_dist1
    avg_ratio2 = np.sqrt(2)/avg_dist2

    normalized_src_points *= avg_ratio1
    normalized_dst_points *= avg_ratio2
    
    normalized_src_points[:, 2] = 1
    normalized_dst_points[:, 2] = 1
    
    T1 = np.array([
        [avg_ratio1, 0, 0],
        [0, avg_ratio1, 0],
        [0,0,1]
    ]) @ np.array([
        [1, 0, -mass_point1[0]],
        [0,1,-mass_point1[1]],
        [0,0,1]
    ])

    T2 = np.array([
        [avg_ratio2, 0, 0],
        [0, avg_ratio2, 0],
        [0,0,1]
    ]) @ np.array([
        [1, 0, -mass_point2[0]],
        [0,1,-mass_point2[1]],
        [0,0,1]
    ])

    return normalized_src_points, normalized_dst_points, T1, T2

def DLT(pts1, pts2):
    NUMB = pts1.shape[0]
    A = np.zeros((2*NUMB,9))
    for i in range(NUMB):
        x1 = pts1[i,0]
        y1 = pts1[i,1]

        x2 = pts2[i,0] 
        y2 = pts2[i,1]

        A[2*i, :] = [-x1, -y1, -1.0, 0, 0, 0, x2*x1, x2*y1, x2]
        A[2*i+1, :] = [0, 0, 0, -x1, -y1, -1.0, y2*x1, y2*y1, y2]

    _,_,V = np.linalg.svd(A)
    V = V.T.conj()
    h = V[:,8]
    H = [[h[0],h[1],h[2]],[h[3],h[4],h[5]],[h[6],h[7],h[8]]]
    return H


def normalizedDLT(pts1, pts2):
    normalized_src_points, normalized_dst_points, T1, T2 = normalizePoints(pts1,pts2)
    H = DLT(normalized_src_points, normalized_dst_points)
    H = np.linalg.inv(T2) @ H @ T1
    return H


def calculateData(normalized_src_points, normalized_dst_points):
    #  % The coordinates of the first point in the first image      
    u11 = normalized_src_points[0, 0]
    v11 = normalized_src_points[0, 1]
    # % The coordinates of the first point in the second image
    u21 = normalized_dst_points[0, 0]
    v21 = normalized_dst_points[0, 1]
    # % The coordinates of the second point in the first image    
    u12 = normalized_src_points[1, 0]
    v12 = normalized_src_points[1, 1]
    # % The coordinates of the second point in the second image
    u22 = normalized_dst_points[1, 0]
    v22 = normalized_dst_points[1, 1]

    # % The SIFT scale of the first point in the first image     
    q11 = M[indices[0], 6]
    # % The SIFT scale of the first point in the second image
    q21 = M[indices[0], 7]
    # % The SIFT rotation angle of the first point in the first image
    a11 = M[indices[0], 8]
    # % The SIFT rotation angle of the first point in the second image
    a21 = M[indices[0], 9]
    # % The SIFT scale of the second point in the first image
    q12 = M[indices[1], 6]
    # % The SIFT scale of the second point in the second image
    q22 = M[indices[1], 7]
    # % The SIFT rotation angle of the second point in the first image
    a12 = M[indices[1], 8]
    # % The SIFT rotation angle of the second point in the second image
    a22 = M[indices[1], 9]

    # % The sines and cosines of the rotation angles
    s11 = np.sin(a11)
    c11 = np.cos(a11)
    s21 = np.sin(a21)
    c21 = np.cos(a21)

    s12 = np.sin(a12)
    c12 = np.cos(a12)
    s22 = np.sin(a22)
    c22 = np.cos(a22)


    # % Compute the null space of the coefficient matrix formed from the
    # % linear equations
    A = np.zeros((6, 9))
    A[0, :]= [0, 0, 0, u11, v11, 1, -u11 * v21, -v11 * v21, -v21]
    A[1, :]= [u11, v11, 1, 0, 0, 0, -u11 * u21, -v11 * u21, -u21]
    A[2, :]= [0, 0, 0, u12, v12, 1, -u12 * v22, -v12 * v22, -v22]
    A[3, :]= [u12, v12, 1, 0, 0, 0, -u12 * u22, -v12 * u22, -u22]
    A[4, :]= [-s21 * c11, -s11 * s21, 0, c11 * c21, s11 * c21, 0, u21 * s21 * c11 - v21 * c11 * c21, u21 * s11 * s21 - v21 * s11 * c21, 0]
    A[5, :]= [-s22 * c12, -s12 * s22, 0, c12 * c22, s12 * c22, 0, u22 * s22 * c12 - v22 * c12 * c22, u22 * s12 * s22 - v22 * s12 * c22, 0]
    n = null_space(A)
    data = np.zeros(49)
    
    data[0:9] = n[:,0]
    data[9:18] = n[:,1]
    data[18:27] = n[:,2]

    # % Coordinates of the first point correspondence
    data[27] = u11
    data[28] = v11
    data[29] = u21
    data[30] = v21

    # % Coordinates of the second point correspondence 
    data[31] = u12
    data[32] = v12
    data[33] = u22
    data[34] = v22

    # % SIFT parameters of the first correspondence
    data[35] = q11
    data[36] = q21
    data[37] = s11
    data[38] = c11
    data[39] = s21
    data[40] = c21

    # % SIFT parameters of the second correspondence  
    data[41] = q12
    data[42] = q22
    data[43] = s12
    data[44] = c12
    data[45] = s22
    data[46] = c22

    # % The normalizing scale
    k1 = normalizing_scale
    k2 = normalizing_scale
    data[47] = 1 / k1
    data[48] = 1 / k2


    return n,data 






if __name__ == "__main__":

    img1 = cv2.imread("data/adamA.png")
    img2 = cv2.imread("data/adamB.png")
    M = np.loadtxt('data/adam.pts')
    N = M.shape[0]

    source_points = M[:, 0:3].T
    destination_points = M[:,3:6].T
    source_points.shape, destination_points.shape

    # % The name of the test scene
    test= 'adam'
    # % The used inlier-outlier threshold in pixels
    threshold= 3.0
    # % The used inlier-outlier threshold in pixels
    confidence= 0.9999
    # % The maximum number of iterations in RANSAC
    iteration_limit= 5e3
    # % The adaptively updated maximum iteration number in RANSAC
    max_iterations= iteration_limit
    # % The truncated threshold for MSAC scoring
    truncated_threshold = threshold * 3 / 2


    best_inliers = []
    best_homography = []
    best_score = 0


    for iter_ in range(int(iteration_limit)):
        indices = np.random.choice(N,2,replace=False)
        
        # indices = np.array([26,2])
        sample_src_points = M[indices, 0:3]
        sample_dst_points = M[indices, 3:6]
        #  % Calculating the normalizing transformation and normalizing
        #     % the point coordinates for estimating the homography.
        
        normalized_src_points, normalized_dst_points, T1, T2 = normalizePoints(sample_src_points, sample_dst_points)

        # % Scale difference between the two normalizing transformations.
        #     % This is requires to normalize the angles and scales
        normalizing_scale = T2[0,0]/T1[0,0] 

        if np.any(np.isnan(normalized_src_points)) or np.any(np.isnan(normalized_dst_points)):
        
            continue
    
        n,data = calculateData(normalized_src_points,normalized_dst_points)
        
        Hs = solverHomography2SIFT(data)
        min_err = 1e10
        sol_number = 10
        best_homography = []

        for i in range(Hs.shape[1]):
            alpha = Hs[0,i]
            beta = Hs[1,i]

            if np.abs(np.imag(alpha)) > 0 or np.abs(np.imag(beta)) > 0:
                continue
            
                #    % Recovering the homography from alpha and beta and
                #         % the null-vectors.

            h = alpha * n[:,0] + beta * n[:,1] + n[:,2]
            Hi = np.linalg.inv(T2) @ np.reshape(h,(3,3)) @ T1

            # Calculate teh score of homography
            pts2_t = Hi @ source_points
            pts2_t = pts2_t / pts2_t[2,:] 
            residuals = np.linalg.norm(destination_points-pts2_t,axis=0)
            inliers = np.where(residuals < truncated_threshold)
            score = np.sum(1 - residuals[inliers] / truncated_threshold)
            
            if score > best_score:
                best_score = score 
                best_homography = Hi
                best_inliers = inliers

                inlier_number = np.count_nonzero(inliers) 
                max_iterations = np.log(1-confidence) / np.log(1-(inlier_number/N)**2)
        if iter_ > max_iterations:
            break

    tpts_1,tpts_2 = M[best_inliers,0:3],M[best_inliers,3:6]
    tpts_1,tpts_2 = tpts_1.reshape((tpts_1.shape[1],tpts_1.shape[2])),tpts_2.reshape((tpts_2.shape[1],tpts_2.shape[2]))
    H = normalizedDLT(tpts_1,tpts_2)
    pts2_t = H @ source_points
    pts2_t = pts2_t / pts2_t[2,:] 
    residuals = np.linalg.norm(destination_points-pts2_t,axis=0)
    best_inliers = np.where(residuals < truncated_threshold)

    
    print("Average Reprojection Error: ", np.mean(residuals[best_inliers]),"px")
    print("Inlier Number: ",len(best_inliers[0]))
    print("Iteration number required for RANSAC: ", iter_)

    
    img = np.column_stack([img1,img2])    
    for i in range(len(best_inliers[0])):
        color =np.random.randint(0,256,3,dtype=int)
        color = (int(color[0]),int(color[1]),int(color[2]))
        cv2.circle(img, (int(M[best_inliers[0][i],0]),int(M[best_inliers[0][i],1])),3,color,2)
        cv2.circle(img, (int(img1.shape[1]+M[best_inliers[0][i],3]),int(M[best_inliers[0][i],4])),3,color,2)
        x1, y1 = int(M[best_inliers[0][i], 0]), int(M[best_inliers[0][i], 1])
        x2, y2 = int(img1.shape[1] + M[best_inliers[0][i], 3]), int(M[best_inliers[0][i], 4])
        cv2.line(img, (x1,y1), (x2,y2),color,1)

        
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


           