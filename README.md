# Joint-Bilateral-Upsampling

In this projetc, bilateral joint guided filter is utilized to perform upsampling on downsampled depth images.

The algorithms are ran on different images located in each direcory.In each folder, the depth map and the RGB image are located, as well ass the downsampled depth map with factor 4 and 8.

RGB image:

![view1](https://user-images.githubusercontent.com/72257286/144595384-a9172e72-94ba-4020-a5ee-83a7adfd0486.png)

Depth map:


![disp1](https://user-images.githubusercontent.com/72257286/144595420-ff1947c3-66e7-4a89-a572-9f84c5d4919a.png)


Downsampled images:

with factor 4:


![downsampled4](https://user-images.githubusercontent.com/72257286/144595466-4e5ae779-5593-4f05-9561-517097f39ce9.png)



with factor 8 :

![downsampled8](https://user-images.githubusercontent.com/72257286/144595489-c05ceeaa-01b0-4aca-b39e-3d7728afbdbc.png)


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

![upsampled Disparity with factor 4](https://user-images.githubusercontent.com/72257286/144597234-822b5760-e678-4bdb-9d94-c6a1e0320dad.png)


factor 8 with area relatoin:


![upsampled Disparity with factor 8 arae relatoin interpolation ](https://user-images.githubusercontent.com/72257286/144597283-a9a1bb26-787d-4a9f-8213-00ee60267ab6.png)


Factor 8 with bicubic interpolation:


![upsampled Disparity with factor 8 bicubic interpolation ](https://user-images.githubusercontent.com/72257286/144597318-e991e1a7-69e4-4802-b7db-c96634a56412.png)


Factor 8 with linear interpolation:

![upsampled Disparity with factor 8 linear interpolation ](https://user-images.githubusercontent.com/72257286/144597381-c16ad728-bc0e-4f53-b590-764c4979fd94.png)

Factor 8 with nearest neighbor interpolation:

![upsampled Disparity with factor 8 nearest neighbor interpolation ](https://user-images.githubusercontent.com/72257286/144598354-b4bfc9e4-8c9e-44fe-81f6-501882edcc56.png)


factor 8 JBU upsampling:


![upsampled Disparity with factor 8](https://user-images.githubusercontent.com/72257286/144598390-36be385f-97aa-4779-ab7b-4a2123a39578.png)

as it can be opbserverd from the results, the JBU methode does not perform good i the case of images downsampled with high factor which migh be the result of applying the bluring iteratively in each iteration.

