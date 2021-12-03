#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <sstream>




cv::Mat CreateGaussianKernel(int window_size, const double sigma) {
	cv::Mat kernel(cv::Size(window_size, window_size), CV_32FC1);

	int half_window_size = window_size / 2;

	// see: lecture_03_slides.pdf, Slide 13
	//const double k = 2.5;
   //const double r_max = std::sqrt(2.0 * half_window_size * half_window_size);
   //const double sigma = r_max / k;

	// sum is for normalization 
	float sum = 0.0;
	double sigma_squared = sigma * sigma;
	for (int x = -window_size / 2; x <= window_size / 2; x++) {
		for (int y = -window_size / 2; y <= window_size / 2; y++) {
			float val = exp(-(x * x + y * y) / (2 * sigma_squared));
			kernel.at<float>(x + window_size / 2, y + window_size / 2) = val;
			sum += val;
		}
	}

	// normalising the Kernel 
	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j)
			kernel.at<float>(i, j) /= sum;

	// note that this is a naive implementation
	// there are alternative (better) ways
	// e.g. 
	// - perform analytic normalisation (what's the integral of the gaussian? :))
	// - you could store and compute values as uchar directly in stead of float
	// - computing it as a separable kernel [ exp(x + y) = exp(x) * exp(y) ] ...
	// - ...

	return kernel;
}



cv::Mat Bilateral(const cv::Mat& input, const int window_size = 5, float sigmaSpatial = 1, float sigmaSpectral = 5) {
	const auto width = input.cols;
	const auto height = input.rows;
	cv::Mat output(input.size(), input.type());

	cv::Mat gaussianKernel = CreateGaussianKernel(window_size, sigmaSpatial); // sigma for the spatial filter (Gaussian, \(w_G\) kernel)

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	auto d = [](float a, float b) {
		return std::abs(a - b);
	};

	auto p = [](float val, float sigma) { // use of weighting function p : dissimilar pixels get lower weights, preserves strong edges, smooths other regions
		const float sigmaSq = sigma * sigma;
		const float normalization = std::sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {

			float sum_w = 0;
			float sum = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {

					float range_difference
						= d(input.at<uchar>(r, c), input.at<uchar>(r + i, c + j));

					float w
						= p(range_difference, sigmaSpectral) // spectral filter
						* gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2); // spatial filter

					sum
						+= input.at<uchar>(r + i, c + j) * w;
					sum_w
						+= w;
				}
			}

			output.at<uchar>(r, c) = sum / sum_w;

		}
	}
	return output;
}

void JointBilateral(const cv::Mat& input, const cv::Mat& guide, cv::Mat& output, const int window_size = 5, const float sigmaSpectral = 20.0, const double sigmaSpatial = 0.23) {
	
	const auto width = input.cols;
	const auto height = input.rows;

	cv::Mat gaussianKernel = CreateGaussianKernel(window_size, sigmaSpatial); 
	auto d = [](float a, float b) {
		return std::abs(a - b);
	};

	auto p = [](float val, float sigma) {	
		const float sigmaSq = sigma * sigma;
		const float normalization = std::sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {

			float sum_w = 0;
			float sum = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {

					float range_difference
						= d(input.at<uchar>(r, c), input.at<uchar>(r + i, c + j)); 

					float w
						= p(range_difference, sigmaSpectral) 
						* gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);

					sum
						+= guide.at<uchar>(r + i, c + j) * w;
					sum_w
						+= w;
				}
			}

			output.at<uchar>(r, c) = sum / sum_w;

		}
	}
}

cv::Mat Upsample(const cv::Mat& input, const cv::Mat& guide, const int window_size = 5, const float sigmaSpectral = 20.0, const double sigmaSpatial = 0.23) {
	int inputHeight = input.rows;
	int guideHeight = guide.rows;
	int upsamplingFactor = log2(inputHeight / guideHeight);
	cv::Mat D = guide.clone(); 
	cv::Mat I = input.clone();
	for (int i = 1; i <= upsamplingFactor -1; ++i)
	{
		cv::resize(D, D, D.size() * 2); 
		cv::resize(input, I, D.size());	
		JointBilateral(I, D, D, window_size,sigmaSpectral, sigmaSpatial); 
	}
	cv::resize(D, D, input.size());
	JointBilateral(input, D, D, window_size, sigmaSpectral, sigmaSpatial); 
	return D;
}

double SSD(const cv::Mat& img1, const cv::Mat& img2)
{
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	return ssd;
}

double RMSE(const cv::Mat& img1, const cv::Mat& img2)
{
	int size = img1.rows * img1.cols;
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	return sqrt(mse);
}

double MSE(const cv::Mat& img1, const cv::Mat& img2)
{
	int size = img1.rows * img1.cols;
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	return mse;
}

double PSNR(const cv::Mat& img1, const cv::Mat& img2)
{

	double max = 255;
	int size = img1.rows * img1.cols;
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	double psnr = 10 * log10((max * max) / mse);
	return psnr;
}

long double mean(const cv::Mat& img)
{
	long double sum = 0;
	int size = img.rows * img.cols;
	for (int r = 0; r < img.rows; ++r) {
		for (int c = 0; c < img.cols; ++c) {
			sum += img.at<uchar>(r, c);
		}
	}
	return sum / size;

}

long double variance(const cv::Mat& img)
{
	cv::Mat var_matrix = img.clone();
	long double sum = 0;
	int size = var_matrix.rows * var_matrix.cols;
	long double mean_ = mean(var_matrix);

	for (int r = 0; r < var_matrix.rows; ++r) {
		for (int c = 0; c < var_matrix.cols; ++c) {
			var_matrix.at<uchar>(r, c) -= mean_;
			var_matrix.at<uchar>(r, c) *= var_matrix.at<uchar>(r, c);
		}
	}

	for (int r = 0; r < var_matrix.rows; ++r) {
		for (int c = 0; c < var_matrix.cols; ++c) {
			sum += var_matrix.at<uchar>(r, c);
		}
	}
	return sum / size;
}

double covariance(const cv::Mat& img1, const cv::Mat& img2)
{
	int size = img1.rows * img1.cols;
	long double sum = 0;
	long double mean1 = mean(img1);
	long double mean2 = mean(img2);
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			sum = sum + ((img1.at<uchar>(r, c) - mean1) * (img2.at<uchar>(r, c) - mean2));
		}
	}
	return sum / size;
}

long double SSIM(const cv::Mat& img1, const cv::Mat& img2)
{
	long double ssim = 0;
	long double k1 = 0.01, k2 = 0.03, L = 255;
	long double C1 = (k1 * L) * (k1 * L);
	long double C2 = (k2 * L) * (k2 * L);

	long double mu_x = mean(img1);
	long double mu_y = mean(img2);
	long double variance_x = variance(img1);
	long double variance_y = variance(img2);
	long double covariance_xy = covariance(img1, img2);

	ssim = ((2 * mu_x * mu_y + C1) * (2 * covariance_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (variance_x * variance_x + variance_y * variance_y + C2));
	return ssim;
}


int main(int argc, char** argv) {

	

	const int windowSize = 7;
	std::string directory = "D:/Users/Parsa/Desktop/Study materials/3D Sensing/practice/hw2/03_bilateral_task_solution/src/data/"; 
	std::string ImageName = "Wood1/";
	cv::Mat input_rgb = cv::imread(directory + ImageName+ "view1.png", 0);
	//creating the downsampled images
	cv::Mat down4, down8;
	cv::resize(input_rgb, down4, input_rgb.size() / 4, 1, 1, cv::INTER_LINEAR);
	cv::resize(input_rgb, down8, input_rgb.size() / 8, 1, 1, cv::INTER_LINEAR);
	cv::imwrite(directory + ImageName + "downsampled4.png", down4);
	cv::imwrite(directory + ImageName + "downsampled8.png", down8);

	cv::Mat input_depth_4 = cv::imread(directory + ImageName+ "downsampled4.png", 0);
	cv::Mat input_depth_8 = cv::imread(directory + ImageName + "downsampled8.png", 0);
	std::ofstream file;
	file.open(directory +ImageName+ "runtime log.txt");
	std::string line;

	


	//cv::Mat im = cv::imread(directory + ImageName + "view1.png", 0);

	if (input_rgb.data == nullptr) {
		std::cerr << "Failed to load image" << std::endl;
	}
	cv::Mat input = input_rgb.clone();
	cv::Mat noise(input_rgb.size(), input_rgb.type());
	uchar mean = 0;
	uchar stddev = 25;
	cv::randn(noise, mean, stddev);
	input_rgb += noise;
	
	//applying the iterative upsampling

	auto t_begin = std::chrono::high_resolution_clock::now();
	cv::Mat upSampled4 = Upsample(input_rgb, input_depth_4, windowSize,1000.0,0.13);
	auto t_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_begin).count();
	line = "upsampling the depth map with 4-factor with JBU methode took " + std::to_string(duration) +" second\n";
	file << line;
	imwrite(directory +ImageName +"/upsample/"+ "upsampled Disparity with factor 4.png", upSampled4);


	t_begin = std::chrono::high_resolution_clock::now();
	cv::Mat upSampled8 = Upsample(input_rgb, input_depth_8, windowSize, 1.0, 0.009);
	t_end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_begin).count();
	line = "upsampling the depth map with 8-factor with JBU methode took " + std::to_string(duration) + " second\n";
	file << line;
	imwrite(directory + ImageName + "/upsample/" + "upsampled Disparity with factor 8.png", upSampled8);
	
	//upsampling using other methosdes for upsampling
	cv::Mat inter_linear, inter_nearest, inter_area, inter_cubic;

	//upsampling using linear interpolatoin
	t_begin = std::chrono::high_resolution_clock::now();
	cv::resize(input_depth_4, inter_linear, input_rgb.size(), 8, 8, cv::INTER_LINEAR);
	t_end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_begin).count();

	double ssd = SSD(input_rgb, inter_linear);
	double rmse = RMSE(input_rgb, inter_linear);
	double psnr = PSNR(input_rgb, inter_linear);
	long double ssim = SSIM(input_rgb, inter_linear);

	line = "upsampling the depth map with 8-factor with linear interpolation methode took " + std::to_string(duration) + " second with "
		+  " SSD = "+ std::to_string(ssd)+ " RMSE = " + std::to_string(rmse) + " PSNR = " + std::to_string(psnr)
		+ " SSIM = " + std::to_string(ssim) +" \n";
	file << line;
	imwrite(directory + ImageName + "/upsample/" + "upsampled Disparity with factor 8 linear interpolation .png", inter_linear);
	
	
	//upsampling using  nearest neighbor interpolation
	t_begin = std::chrono::high_resolution_clock::now();
	cv::resize(input_depth_4, inter_nearest, input_rgb.size(), 8, 8, cv::INTER_NEAREST);
	t_end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_begin).count();

	ssd = SSD(input_rgb, inter_nearest);
	rmse = RMSE(input_rgb, inter_nearest);
	psnr = PSNR(input_rgb, inter_nearest);
	ssim = SSIM(input_rgb, inter_nearest);

	line = "upsampling the depth map with 8-factor with nearest neighbor interpolation methode took " + std::to_string(duration) + " second with "
		+ " SSD = " + std::to_string(ssd) + " RMSE = " + std::to_string(rmse) + " PSNR = " + std::to_string(psnr)
		+ " SSIM = " + std::to_string(ssim) + " \n";
	file << line;
	imwrite(directory + ImageName + "/upsample/" + "upsampled Disparity with factor 8 nearest neighbor interpolation .png", inter_linear);

	
	//upsampling using  interpolatoni using area relatoin
	
	t_begin = std::chrono::high_resolution_clock::now();
	cv::resize(input_depth_4, inter_area, input_rgb.size(), 8, 8, cv::INTER_AREA);
	t_end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_begin).count();

	ssd = SSD(input_rgb, inter_area);
	rmse = RMSE(input_rgb, inter_area);
	psnr = PSNR(input_rgb, inter_area);
	ssim = SSIM(input_rgb, inter_area);

	line = "upsampling the depth map with 8-factor with area relation interpolation methode took " + std::to_string(duration) + " second with "
		+ " SSD = " + std::to_string(ssd) + " RMSE = " + std::to_string(rmse) + " PSNR = " + std::to_string(psnr)
		+ " SSIM = " + std::to_string(ssim) + " \n";
	file << line;
	imwrite(directory + ImageName + "/upsample/" + "upsampled Disparity with factor 8 arae relatoin interpolation .png", inter_linear);

	
	//upsampling using  bicubic interpolatoin

	t_begin = std::chrono::high_resolution_clock::now();
	cv::resize(input_depth_4, inter_cubic, input_rgb.size(), 8, 8, cv::INTER_CUBIC);
	t_end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_begin).count();

	ssd = SSD(input_rgb, inter_cubic);
	rmse = RMSE(input_rgb, inter_cubic);
	psnr = PSNR(input_rgb, inter_cubic);
	ssim = SSIM(input_rgb, inter_cubic);

	line = "upsampling the depth map with 8-factor with bicubic interpolation methode took " + std::to_string(duration) + " second with "
		+ " SSD = " + std::to_string(ssd) + " RMSE = " + std::to_string(rmse) + " PSNR = " + std::to_string(psnr)
		+ " SSIM = " + std::to_string(ssim) + " \n";
	file << line;
	imwrite(directory + ImageName + "/upsample/" + "upsampled Disparity with factor 8 bicubic interpolation .png", inter_linear);



	file.close();
	std::vector<float> sigmaSpatial = { 0.8, 1.2, 2.0, 3.2 }; 
	std::vector<float> sigmaSpectral = { 100., 10000., 500000., 1000000.0 };
	for (int i = 0; i < 4; i++) {	
		for (int j = 0; j < 4; j++) {
			cv::Mat output = Bilateral(input_rgb, windowSize, sigmaSpectral[i], sigmaSpatial[j]);
			double ssd = SSD(input, output);
			double rmse = RMSE(input, output);
			double psnr = PSNR(input, output);
			long double ssim = SSIM(input, output);
			std::string outputName= directory + ImageName +"/bilateral/"+" spectral sigma  "+ std::to_string(sigmaSpectral[i])  +"spatial sigma " + std::to_string(sigmaSpatial[j])
				+" SSIM "+ std::to_string(ssim) + " PSNR " + std::to_string(psnr) +
				" RMSE " + std::to_string(rmse) + " SSD " + std::to_string(ssd) + ".png";
			cv::imwrite(outputName,output);
		}
	}
	
	return 0;
}