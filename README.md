# Local Binary Patterns:
Local Binary Patterns (LBP) is a concept that was first described in 1994 and later became an important feature to classify texture. In our project LBP serves as a feature to represent facial images. When combined with Histogram of Oriented Gradients, the efficiency is greatly increased. Using this combination, each facial image is represented by a simple data vector. The advantages of using LBP include its computational simplicity and its discriminative power. This makes it fast and robust to illumination changes in grayscale images.

To calculate the **Original LBP** representation of an image, the following steps are involved:
1. A sliding window is used where a square window of fixed size is moved over the image and the value of the centre pixel is calculated according to its neighbours.
2. In one square window, the value of the centre pixel is assumed as a threshold.
3. If the value of a neighbouring pixel is greater than the threshold, it is assumed as 1 and if it is lesser, it is assumed as 0.
4. Hence considering a 3x3 window, 8 neighbours give 8 binary values which is then converted into its corresponding decimal value. The order of the binary number is usually taken row wise but not limited to it.
5. The centre pixel value in the LBP image is the newly obtained decimal number. Therefore the entire picture is converted after the sliding window covers all the pixels.

Extended LBP also known as Circular LBP. The original LBP procedure was expanded to incorporate different parameters such as neighbours and radius. Hence instead of a square window, a circular frame is used. The radius defines how far away the neighbours are from the centre pixel. The neighbours define the number of pixels to be considered on the circle's circumference. In this method, it is obvious that all neighbouring points may not lie perfectly on a pixel. Hence to find the value of a data point, values from the 4 nearest pixels are used. Bi-linear Interpolation is used to calculate one value from the 4 pixel values and its distance from the data point.
Once the values are obtained, the circle is converted to binary either clockwise or anti-clockwise(must be same throughout) and centre pixel value of LBP image is found.
The efficiency corresponding to different parameters in our project is shown in results.

Histogram of Oriented Gradients
Through the conversion of an image to its LBP form, facial features are extracted in a way that computers can differentiate better. But to compare two LBP images is equivalent to comparing two normal images in terms of computational complexity and time taken. Hence it is combined with a concept known as Histogram of Oriented Gradients which is known not only to make comparison easier but also improve face discriminating power.
he histogram is most commonly used to show frequency distributions. In this concept, each histogram consists of 256 bars, each for one pixel intensity since grayscale images have pixel intensities between 0 and 255. Each bar shows the frequency of that pixel intensity in the picture.

To conserve feature according to their position in the image, spatial histograms are used. This involves splitting the LBP image into grids and computing the histogram of each grid and concatenating all the histograms in the end. The grids can be of different sizes and the efficiency of our model corresponding to a few sizes are shown in the results.
Therefore, each face image is converted to an array of values. This feature extraction serves as the base for our facial recognition and emotion identification processes.
