import cv2
import numpy as np


def original_lbp(gray_image):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgLBP = np.zeros_like(gray_image)
    neighbours = 3
    #print("Starting LBP")
    for ih in range(0, gray_image.shape[0]-neighbours):
        for iw in range(0, gray_image.shape[1]-neighbours):
            #print("running LBP")
            img = gray_image[ih:ih+neighbours, iw:iw+neighbours]
            center = img[1, 1]
            img01 = (img >= center)*1.0
            img01_vector = img01.T.flatten()
            img01_vector = np.delete(img01_vector, 4)
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2 ** where_img01_vector)
            else:
                num = 0
            imgLBP[ih + 1, iw + 1] = num
    #print("LBP done")
    return imgLBP


def only_uniform(image):
    uniform = [0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58,
                13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 17, 58, 58, 58, 58, 58,
                58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                58, 58, 58, 58, 58, 58, 24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, 29, 30, 58, 31,
                58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                58, 34, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
                58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40, 58, 58,
                58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41, 42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58,
                58, 58, 58, 46, 47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57]
    result = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i][j] = uniform[image[i][j]]
    return result


def plot_hist(LBP_image, grid_row=8, grid_col=8):
    img_height, img_width = LBP_image.shape
    nox = int(np.floor(img_width / grid_col))
    noy = int(np.floor(img_height / grid_row))
    hist = []
    for row in range(grid_row):
        for col in range(grid_col):
            curr = LBP_image[row*noy:(row+1)*noy, col*nox:(col+1)*nox]
            histo, bin_edges = np.histogram(curr, bins=256)
            hist.extend(histo)
    return np.asarray(hist)


def extended_lbp(img):
    img = np.asanyarray(img)
    ysize, xsize = img.shape
    neighbours = 8
    radius = 1
    angles = 2 * np.pi / neighbours
    theta = np.arange(0, 2 * np.pi, angles)
    sample_points = np.array([-np.sin(theta), np.cos(theta)]).T
    sample_points *= radius
    miny = min(sample_points[:, 0])
    maxy = max(sample_points[:, 0])
    minx = min(sample_points[:, 1])
    maxx = max(sample_points[:, 1])
    blocksizey = np.ceil(max(maxy, 0)) - np.floor(min(miny, 0)) + 1
    blocksizex = np.ceil(max(maxx, 0)) - np.floor(min(minx, 0)) + 1
    origy = int(0 - np.floor(min(miny, 0)))
    origx = int(0 - np.floor(min(minx, 0)))
    dx = int(xsize - blocksizex + 1)
    dy = int(ysize - blocksizey + 1)
    C = np.asarray(img[origy:origy + dy, origx:origx + dx], dtype=np.uint8)
    result = np.zeros((dy, dx), dtype=np.uint8)
    for i, p in enumerate(sample_points):
        y, x = p + (origy, origx)
        fx = int(np.floor(x))
        fy = int(np.floor(y))
        cx = int(np.ceil(x))
        cy = int(np.ceil(y))
        ty = y - fy
        tx = x - fx
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty
        N = w1 * img[fy:fy + dy, fx:fx + dx]
        np.add(N, w2 * img[fy:fy + dy, cx:cx + dx], out=N, casting="unsafe")
        np.add(N, w3 * img[cy:cy + dy, fx:fx + dx], out=N, casting="unsafe")
        np.add(N, w4 * img[cy:cy + dy, cx:cx + dx], out=N, casting="unsafe")
        D = N >= C
        np.add(result, (1 << i) * D, out=result, casting="unsafe")
    return result


def hist_compare(hist1, hist2, metric):
    hist1 = np.asarray(hist1).flatten()
    hist2 = np.asarray(hist2).flatten()
    if metric == 1:
        #Euclidean distance
        return np.sqrt(np.sum(np.power((hist1 - hist2), 2)))
    elif metric == 2:
        #Chi-square distance
        bin_dists = (hist1 - hist2) ** 2 / (hist1 + hist2 + np.finfo('float').eps)
        return np.sum(bin_dists)
    elif metric == 3:
        return -np.dot(hist1.T, hist2) / (np.sqrt(np.dot(hist1, hist1.T) * np.dot(hist2, hist2.T)))
    else:
        #Normalized correlation
        pmu = hist1.mean()
        qmu = hist2.mean()
        pm = hist1 - pmu
        qm = hist2 - qmu
        return 1.0 - (np.dot(pm, qm) / (np.sqrt(np.dot(pm, pm)) * np.sqrt(np.dot(qm, qm))))


if __name__ == '__main__':
    pass