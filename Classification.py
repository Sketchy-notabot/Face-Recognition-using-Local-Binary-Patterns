from LBPandHist import *
from Preprocessing import *


def predict(test_img, metric=2):
    k = 3
    labels = np.loadtxt('labeldata.csv', dtype=str, delimiter=',')
    histograms = np.loadtxt('histodata.csv', delimiter=',')
    roi_face_ = preprocess(test_img)
    LBP_image_ = extended_lbp(roi_face_)
#   LBP_image_ = only_uniform(LBP_image_)
    test_hist = plot_hist(LBP_image_)
    k_neighbors = []
    dist = hist_compare(test_hist, histograms[0], metric)
    max_tuple = (0, dist)
    k_neighbors.append(max_tuple)
    for i in range(1, len(labels)):
        curr_dist = hist_compare(test_hist, histograms[i], metric)
        if len(k_neighbors) < k:
            if curr_dist > max_tuple[1]:
                max_tuple = (i, curr_dist)
            k_neighbors.append((i, curr_dist))
        else:
            if curr_dist < max_tuple[1]:
                tuple_del = max_tuple
                max_tuple = (i, curr_dist)
                k_neighbors.append(max_tuple)
                for j in range(k):
                    if k_neighbors[j] == tuple_del:
                        k_neighbors.pop(j)
                    else:
                        if max_tuple[1] < k_neighbors[j][1]:
                            max_tuple = k_neighbors[j]
    freq = {}
    min_dist = k_neighbors[0][1]
    min_index = k_neighbors[0][0]
    for i in range(k):
        if i == 0:
            pass
        else:
            if min_dist > k_neighbors[i][1]:
                min_index = k_neighbors[i][0]
        if labels[k_neighbors[i][0]] not in freq:
            freq[labels[k_neighbors[i][0]]] = 1
        else:
            freq[labels[k_neighbors[i][0]]] += 1

    if len(freq) == k:
        return labels[min_index]
    else:
        return max(freq, key=freq.get)


if __name__ == '__main__':
    pass

