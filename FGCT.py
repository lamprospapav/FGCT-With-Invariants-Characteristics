import cv2
import numpy as np
import tri_area
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


def extract_triangles(testImages, logoImages, pairs, num_tri):
    if len(set(pairs)) < 10:
        Dab, a, b = [1, 1, 1]
        return Dab, a, b

    if len(pairs) < 300:
        max_area = 800
        attemps = 300
    else:
        max_area = 800
        attemps = 600

    x = np.random.randint(0, len(pairs), size=(num_tri, 3))
    id_logo = np.zeros((num_tri, 3), int)
    id_im = np.zeros((num_tri, 3), int)

    for idx, val in enumerate(x):
        id_logo[idx] = [pairs[val[0]].trainIdx, pairs[val[1]].trainIdx, pairs[val[2]].trainIdx]
        id_im[idx] = [pairs[val[0]].queryIdx, pairs[val[1]].queryIdx, pairs[val[2]].queryIdx]

    id_logo.sort(axis=1)
    id_im.sort(axis=1)
    a = []
    b = []
    A = np.zeros((num_tri, 1), float)
    B = np.zeros((num_tri, 1), float)
    A_angles = []
    B_angles = []
    angles = np.zeros((num_tri, 1), int)

    for idx in range(num_tri):
        b.append(np.transpose([[testImages[2][id_im[idx][0]].pt[0], testImages[2][id_im[idx][0]].pt[1], 1],
                               [testImages[2][id_im[idx][1]].pt[0], testImages[2][id_im[idx][1]].pt[1], 1],
                               [testImages[2][id_im[idx][2]].pt[0], testImages[2][id_im[idx][2]].pt[1], 1]]))
        a.append(np.transpose([[logoImages[2][id_logo[idx][0]].pt[0], logoImages[2][id_logo[idx][0]].pt[1], 1],
                               [logoImages[2][id_logo[idx][1]].pt[0], logoImages[2][id_logo[idx][1]].pt[1], 1],
                               [logoImages[2][id_logo[idx][2]].pt[0], logoImages[2][id_logo[idx][2]].pt[1], 1]]))
        A[idx] = tri_area.tri_area(np.copy(a[idx])[0:2, 0], np.copy(a[idx])[0:2, 1], np.copy(a[idx])[0:2, 2])
        B[idx] = tri_area.tri_area(np.copy(b[idx])[0:2, 0], np.copy(b[idx])[0:2, 1], np.copy(b[idx])[0:2, 2])
        A_angles.append(tri_area.tri_angles(np.copy(a[idx])[0:2, 0], np.copy(a[idx])[0:2, 1], np.copy(a[idx])[0:2, 2]))
        B_angles.append(tri_area.tri_angles(np.copy(b[idx])[0:2, 0], np.copy(b[idx])[0:2, 1], np.copy(b[idx])[0:2, 2]))

        if np.any(A_angles[idx] < 10) or np.any(B_angles[idx] < 10):
            angles[idx] = 1
        else:
            angles[idx] = 0

    area_test = 300
    indexes = np.unique(np.concatenate((np.where(A < 10)[0], np.where(B < 10)[0], np.where(A > area_test)[0],
                                        np.where(B > area_test)[0], np.where(angles == 1)[0])))
    accepted_indexes = np.setdiff1d(np.arange(0, num_tri), indexes)
    temp_size = len(pairs)
    accepted_id_logo = np.zeros(num_tri)
    accepted_id_im =np.zeros(num_tri)
    if np.logical_not(isinstance(accepted_indexes,int)):
        for idx, val in enumerate(accepted_indexes):
            accepted_id_logo[idx] = id_logo[val][0]*temp_size**2 +id_logo[val][1]*temp_size + id_logo[val][2]
            accepted_id_im[idx] = id_im[val][0] * temp_size ** 2 + id_im[val][1] * temp_size + id_im[val][2]
    n = 1
    while indexes.size != 0 and area_test < max_area:
        if n > attemps:
            area_test += 50
            n = 1
        x = np.random.randint(0, len(pairs), size=(len(indexes), 3))
        for idx, val in enumerate(x):
            id_logo[indexes[idx]] = [pairs[val[0]].trainIdx, pairs[val[1]].trainIdx, pairs[val[2]].trainIdx]
            id_im[indexes[idx]] = [pairs[val[0]].queryIdx, pairs[val[1]].queryIdx, pairs[val[2]].queryIdx]
            a[indexes[idx]] = (np.transpose(
                [[logoImages[2][id_logo[indexes[idx]][0]].pt[0], logoImages[2][id_logo[indexes[idx]][0]].pt[1], 1],
                 [logoImages[2][id_logo[indexes[idx]][1]].pt[0], logoImages[2][id_logo[indexes[idx]][1]].pt[1], 1],
                 [logoImages[2][id_logo[indexes[idx]][2]].pt[0], logoImages[2][id_logo[indexes[idx]][2]].pt[1], 1]]))
            b[indexes[idx]] = (np.transpose(
                [[testImages[2][id_im[indexes[idx]][0]].pt[0], testImages[2][id_im[indexes[idx]][0]].pt[1], 1],
                 [testImages[2][id_im[indexes[idx]][1]].pt[0], testImages[2][id_im[indexes[idx]][1]].pt[1], 1],
                 [testImages[2][id_im[indexes[idx]][2]].pt[0], testImages[2][id_im[indexes[idx]][2]].pt[1], 1]]))
            A[indexes[idx]] = tri_area.tri_area(np.copy(a[indexes[idx]])[0:2, 0], np.copy(a[indexes[idx]])[0:2, 1],
                                                np.copy(a[indexes[idx]])[0:2, 2])
            B[indexes[idx]] = tri_area.tri_area(np.copy(b[indexes[idx]])[0:2, 0], np.copy(b[indexes[idx]])[0:2, 1],
                                                np.copy(b[indexes[idx]])[0:2, 2])
            A_angles[indexes[idx]] = (
                tri_area.tri_angles(np.copy(a[indexes[idx]])[0:2, 0], np.copy(a[indexes[idx]])[0:2, 1],
                                    np.copy(a[indexes[idx]])[0:2, 2]))
            B_angles[indexes[idx]] = (
                tri_area.tri_angles(np.copy(b[indexes[idx]])[0:2, 0], np.copy(b[indexes[idx]])[0:2, 1],
                                    np.copy(b[indexes[idx]])[0:2, 2]))

            logo_id = id_logo[indexes[idx]][0]*temp_size**2 + id_logo[indexes[idx]][1]*temp_size + id_logo[indexes[idx]][2]
            test_id = id_im[indexes[idx]][0]*temp_size**2 + id_im[indexes[idx]][1]*temp_size + id_im[indexes[idx]][2]



            if np.any(A_angles[indexes[idx]] < 10) or np.any(B_angles[indexes[idx]] < 10)  or np.any([logo_id == val1 for val1 in accepted_id_logo]) or np.any([test_id == val2 for val2 in accepted_id_im]):
                angles[indexes[idx]] = 1
            else:
                angles[indexes[idx]] = 0
        del indexes, x
        indexes = np.unique(np.concatenate((np.where(A < 10)[0], np.where(B < 10)[0], np.where(A > area_test)[0],
                                            np.where(B > area_test)[0], np.where(angles == 1)[0])))
        accepted_indexes = np.setdiff1d(np.arange(0, num_tri), indexes)
        if np.logical_not(isinstance(accepted_indexes, int)):
            for idx, val in enumerate(accepted_indexes):
                accepted_id_logo[idx] = id_logo[val][0] * temp_size ** 2 + id_logo[val][1] * temp_size + id_logo[val][2]
                accepted_id_im[idx] = id_im[val][0] * temp_size ** 2 + id_im[val][1] * temp_size + id_im[val][2]

        n += 1
    if indexes.size != 0:
        a = [a[idx] for idx in np.setdiff1d(np.arange(0, num_tri), indexes)]
        b = [b[idx] for idx in np.setdiff1d(np.arange(0, num_tri), indexes)]

    if len(a) < 5:
        Dab, a, b = [1, 1, 1]
        return Dab, a, b

    R_logo = [[[] for y in range(len(a))] for i in range(len(a))]
    R_test = [[[] for y in range(len(b))] for i in range(len(b))]
    Dab = np.zeros((len(a), len(a)), float)
    for i, (var11, var12) in enumerate(zip(a, b)):
        for j, (var21, var22) in enumerate(zip(a, b)):
            t = np.dot(np.linalg.inv(var11), var21)
            t[t < 0.0001] = 0
            t[np.isinf(t)] = 0
            t = np.nan_to_num(t)
            R_logo[j][i] = t
            t = np.dot(np.linalg.inv(var12), var22)
            t[np.isinf(t)] = 0
            t = np.nan_to_num(t)
            t[t < 0.0001] = 0
            R_test[j][i] = t
            relative_error = np.abs((R_logo[j][i] - R_test[j][i])) / np.abs(R_logo[j][i] + R_test[j][i])
            relative_error = np.nan_to_num(relative_error)
            relative_error[np.isinf(relative_error)] = 0
            relative_error[relative_error < 0.0001] = 0
            if np.sum(R_logo[j][i]) == np.sum(R_test[j][i]) and len(logoImages[2])!=len(testImages[2]):
                relative_error = 1

            Dab[j, i] = np.linalg.norm(relative_error)

    return Dab, a, b


def FGCT(Dab, alpha, sigma):
    if np.all(Dab == 1):
        correspodence1, correspodence2, consistensy, Dtemp = [0, 0, 0, 0]
        return correspodence1, correspodence2, consistensy, Dtemp

    Dab = np.exp((-Dab ** 2) / (sigma ** 2))
    D = Dab + np.transpose(Dab)
    K = np.ones(len(D)) / len(D)
    dk = K

    while np.linalg.norm(dk) > 10 ** (-5):
        G = np.exp(alpha * np.dot(D, K))
        dk = G / np.linalg.norm(G) - K
        K = K + dk

    idx_temp = np.argsort(-K)
    np.fill_diagonal(D, 0)
    D[D==2] =0
    Dtemp = D[idx_temp, :][:, idx_temp]
    consistensy = np.zeros(len(idx_temp))

    for idx in range(len(idx_temp)):
        t = K * 0
        t[idx_temp[0:idx]] = 1 / (idx + 1)
        consistensy[idx] = np.linalg.multi_dot([np.transpose(t), D, t])
    consistensy = -np.sort(-consistensy)
    correspodence1 = (np.sum(consistensy) / len(Dab)) * 100 / 2
    correspodence2 = np.sum(consistensy >= 0.8) / len(Dab) * 100

    return correspodence1, correspodence2, consistensy, Dtemp


def plot_triangles(logoImages, testImages, consistensy, Dtemp, a, b, pairs):
    logo, ax1 = plt.subplots()
    test, ax2 = plt.subplots()
    ax2.set_title('TestImage')
    ax1.set_title('LogoImage')
    ax1.imshow(cv2.cvtColor(logoImages[0], cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(testImages[0], cv2.COLOR_BGR2RGB))

    for val in pairs:
        ax1.scatter(logoImages[2][val.trainIdx].pt[0], logoImages[2][val.trainIdx].pt[1], s=10, c='red', marker='*')
        ax2.scatter(testImages[2][val.queryIdx].pt[0], testImages[2][val.queryIdx].pt[1], s=10, c='red', marker='*')
    if np.logical_not(isinstance(a,int)):
        for val in a:
            ax1.t1 = plt.Polygon(np.transpose(val[:2, :]), color='blue', fill=False)
            ax1.add_patch(ax1.t1)

    logo.show()
    if np.logical_not(isinstance(Dtemp,int)):
        for val in b:
            ax2.t1 = plt.Polygon(np.transpose(val[:2, :]), color="blue", fill=False)
            ax2.add_patch(ax2.t1)

    test.show()
    if isinstance(Dtemp, int):
        Dtemp = np.zeros((50, 50))

    fig3 =plt.figure(3)
    ax1 = fig3.add_subplot(121)
    ax2 =fig3.add_subplot(122)
    ax1.set_title('Consistency')
    ax1.plot(consistensy)
    ax2.set_title('Dtemp')
    ax2.imshow(Dtemp, extent=[0, len(Dtemp), 0, len(Dtemp)])

    fig3.show()
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    plt.close('all')
