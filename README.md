# Joint-Bilateral-Upsampling
 # Parsa Pourzargham 
 # Neptune ID: LUM2CE

In this projetc, bilateral joint guided filter is utilized to perform upsampling on downsampled depth images.

The algorithms are ran on different images located in each direcory.In each folder, the depth map and the RGB image are located, as well ass the downsampled depth map with factor 4 and 8.

RGB image:

![view1](https://user-images.githubusercontent.com/72257286/144595384-a9172e72-94ba-4020-a5ee-83a7adfd0486.png)

Depth map:


![disp1](https://user-images.githubusercontent.com/72257286/144595420-ff1947c3-66e7-4a89-a572-9f84c5d4919a.png)


Downsampled images:

with factor 4:

![downsampled4](https://user-images.githubusercontent.com/72257286/144610737-77861e0c-0022-4ea9-bbac-a91eb0d4e9f2.png)


with factor 8 :

![downsampled8](https://user-images.githubusercontent.com/72257286/144610779-e197da67-40d2-47d2-9962-944d3d87ae23.png)



there are two folders in each directory called Bilateral and upsample.In bilateral folder, 16 different outputs of the bilateral filter with different sigma parameters are located.For each output, the image name contains the sigma values,both spatial and spectral sigma,the similarity measures : SSIM,PSNR,SSD and RMSE .In upsample folder, the upsampled images are located.In each image directory, a text file is also created with the program containing the runtime of various different algorithms like linear interpolatoin, bicubic interpolation, nearest neighbor and area relation based upsampling and their runtime,as well as the similarity measures measured during the runtime.


Bilateral filtering:

16 different sigma parameters are rtested over the RGB image.Noise has been added to each RGB image and then the bilateral filter is applied to the images with the corresponding sigma values.

main RGB image:


![view1](https://user-images.githubusercontent.com/72257286/144596330-da56156c-ce49-4ffb-80ab-ec9c8e5c2110.png)

outputs of bilateral filter:

with spatial sigma=0.8 and spectral sigma=100 :


![spectral sigma  100 000000spatial sigma 0 800000 SSIM 0 253763 PSNR 16 766212 RMSE 37 002375 SSD 235566688 000000](https://user-images.githubusercontent.com/72257286/144596440-73b36e2c-6881-4f87-a37d-af30fdb3e503.png)

with spatial sigma=3.2 and spectral sigma=100000 :



![spectral sigma  1000000 000000spatial sigma 3 200000 SSIM 0 245398 PSNR 17 280169 RMSE 34 876408 SSD 209275396 000000](https://user-images.githubusercontent.com/72257286/144596516-982e9dbc-988f-49c8-bf5f-b86cad4c230d.png)

The runtime of the applied algorithms are:

factor 4 with our JBU :1.57 sec

factor 8 with out JBU :1.05 sec

factor 8 with linear interpolation:0.003 sec

factor 8 with nearest neighbor : 0.0004

factor 8 with area relation : 0.003

factor 8 with bicubic interpolation : 0.004

Similarity measures SSIM, SSD,RMSE and SSD values can be found in the runtime log.txt file ien each folder.

and here are the outputs of each algorithm :

Factor 4 with JBU:

![upsampled Disparity with factor 4](https://user-images.githubusercontent.com/72257286/144610901-118e9cbb-803b-44a1-bcb2-40a96f4c4455.png)


factor 8 with area relatoin:

![upsampled Disparity with factor 8 arae relatoin interpolation ](https://user-images.githubusercontent.com/72257286/144610915-03eeb293-311a-4898-9f5b-cd719d3e5d80.png)

Factor 8 with bicubic interpolation:

![upsampled Disparity with factor 8 bicubic interpolation ](https://user-images.githubusercontent.com/72257286/144610921-0c5137c8-1223-4ce4-b467-a7203b2bc8e8.png)


Factor 8 with linear interpolation:

![upsampled Disparity with factor 8 linear interpolation ](https://user-images.githubusercontent.com/72257286/144610929-fe98a541-b69f-47c3-a2a2-9e89f5af8d87.png)

Factor 8 with nearest neighbor interpolation:

![upsampled Disparity with factor 8 nearest neighbor interpolation ](https://user-images.githubusercontent.com/72257286/144610950-4bdaf16f-d708-4ffb-a5d4-50da0709e6e3.png)


factor 8 JBU upsampling:

![upsampled Disparity with factor 8](https://user-images.githubusercontent.com/72257286/144610960-2cc3e821-83b7-4dbb-b171-df296e250bcb.png)


as it can be opbserverd from the results, the JBU methode does not perform good i the case of images downsampled with high factor which migh be the result of applying the bluring iteratively in each iteration.


figures:

Here is a figure of the runtime of different algorithms(aloe example was used):


![Runtime](https://user-images.githubusercontent.com/72257286/144614568-f8481cf0-5bf5-4320-bd25-b0d2817c1fa4.png)

as it can be observed, our Implementations are significantly more cpu-intense thatn the opencv implemntations, which are noticably faster and more efficient.

the fastest methode was observed to be the nearest neigbor interpolation.

Additional notes:

1- the project is put in a for loop which cmputes the individual images in eacgh forlder fully automatically

2- the datasets used in this project are Middlebury 2005 adn 2006 (18 images in total)

