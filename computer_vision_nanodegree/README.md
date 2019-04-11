# Computer Vision Nanodegree

## Projects
1. [Facial Keypoints Detection](https://github.com/Brandon-HY-Lin/P1_Facial_Keypoints)
	- Purpose: Identify keypoints around eyes, mouth, heads, etc.
	- Framework and Library: Pytorch and OpenCV.
	- Algorithm: CNN + Batch-Norm.
	- [Main Program](https://github.com/Brandon-HY-Lin/P1_Facial_Keypoints/blob/master/3.%20Facial%20Keypoint%20Detection%2C%20Complete%20Pipeline.ipynb)
	- Dataset: [YouTube Faces DB](https://www.cs.tau.ac.il/~wolf/ytfaces/)

2. [Image Captioning](https://github.com/Brandon-HY-Lin/CVND---Image-Captioning-Project)
	- Purpose: Generate caption of a picture with the average BLEU-4 score of 0.517.
	- Framework and Library: Pytorch and TorchVision.
	- Algorithm: CNN + Batch-Norm + Data-Augment.
	- [Main Program](https://github.com/Brandon-HY-Lin/CVND---Image-Captioning-Project/blob/master/2_Training.ipynb)
	- Dataset: [CoCo](http://cocodataset.org/#home) 2014

3. [Landmark Detection & Tracking](https://github.com/Brandon-HY-Lin/P3_Implement_SLAM)
	- Purpose: Sense stationary landmarks using robot's sensor.
	- Algorithm: SLAM.
	- [Main Program](https://github.com/Brandon-HY-Lin/P3_Implement_SLAM/blob/master/3.%20Landmark%20Detection%20and%20Tracking.ipynb)


## Labs
* Part 1: Introduction to Computer Vision
	* Image Representation and Classification
		* [Images as Numerical Data](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/1.%20Images%20as%20Numerical%20Data.ipynb)
			- Purpose: Inspect image format.

		* [Visualizing RGB Channels](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/2.%20Visualizing%20RGB%20Channels.ipynb)
			- Purpose: Display RGB channels.

		* [Blue Screen](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/3.%20Blue%20Screen.ipynb)
			- Purpose: Mask and add a background image

		* [Green Screen](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/4.%20Green%20Screen%20Car.ipynb)

		* [Color Conversion](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/5_1.%20HSV%20Color%20Space%2C%20Balloons.ipynb)

		* [Load and Visualize the Data](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/6_1.%20Visualizing%20the%20Data.ipynb)

		* [Standardizing the Data](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/6_2.%20Standardizing%20the%20Data.ipynb)

		* [Average Brightness](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/6_3.%20Average%20Brightness.ipynb)

		* [Classification](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/6_4.%20Classification.ipynb)
			- Purpose: Day and night Image using gray level counts.

		* [Accuracy and Misclassification](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_1_Image_Representation/6_5.%20Accuracy%20and%20Misclassification.ipynb)
			- Purpose: Calculate the error rate of 'Classification' lab

	* Convolutional Filters and Edge Detector
		* [Fourier Transform](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/1.%20Fourier%20Transform.ipynb)

		* [Finding Edges](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/2.%20Finding%20Edges%20and%20Custom%20Kernels.ipynb)

		* [Gaussian Blur](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/3.%20Gaussian%20Blur.ipynb)

		* [Fourier Transform of Filters](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/4.%20Fourier%20Transform%20of%20Filters.ipynb)

		* [Canny Edge Detection](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/5.%20Canny%20Edge%20Detection.ipynb)

		* [Hough lines](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/6_1.%20Hough%20lines.ipynb)

		* [Haar Cascade and Face Detection](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_2_Convolutional_Filters_Edge_Detection/7.%20Haar%20Cascade%2C%20Face%20Detection.ipynb)

	* Types of Features and Image Segmentation
		* [Find the Corners](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_3_Types_of_Features_Image_Segmentation/1.%20Harris%20Corner%20Detection.ipynb)

		* [Find Contours and Features](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_3_Types_of_Features_Image_Segmentation/2.%20Contour%20detection%20and%20features.ipynb)

		* [K-means Clustering](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_3_Types_of_Features_Image_Segmentation/3.%20K-means.ipynb)

	* Feature Vectors
		* [Image Pyramids](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_4_Feature_Vectors/1.%20Image%20Pyramids.ipynb)

		* [Implementing ORB](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_4_Feature_Vectors/2.%20ORB.ipynb)

		* [Implementing HOG](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_4_Feature_Vectors/3_1.%20HOG.ipynb)

	* CNN Layers and Feature Visualization
		* [Visualizing a Convolutional Layer](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_5_CNN_Layers/1.%20Conv%20Layer%20Visualization.ipynb)

		* [Visualizing a Pooling Layer](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_5_CNN_Layers/2.%20Pool%20Visualization.ipynb)

		* [Visualizing FashionMNIST](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_5_CNN_Layers/3.%20Load%20and%20Visualize%20FashionMNIST.ipynb)

		* [Fashion MNIST Training Exercise](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_5_CNN_Layers/4_1.%20Classify%20FashionMNIST%2C%20exercise.ipynb)

		* [Feature Viz for FashionMNIST](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_5_CNN_Layers/5_1.%20Feature%20viz%20for%20FashionMNIST.ipynb)

		* [Visualize Your Net Layers](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/1_5_CNN_Layers/5_2.%20Visualize%20Your%20Net.ipynb)


* Part 2: Advanced Computer Vision & Deep Learning
	* [YOLO Implementation](https://github.com/Brandon-HY-Lin/CVND_Exercises/tree/master/2_2_YOLO)

	* [LSTM Structure and Hidden State, PyTorch](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/2_4_LSTMs/1.%20LSTM%20Structure.ipynb)

	* [LSTM for Part of Speech Tagging](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/2_4_LSTMs/2.%20LSTM%20Training%2C%20Part%20of%20Speech%20Tagging-Brandon.ipynb)

	* [Character-Level LSTM](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/2_4_LSTMs/3_1.Chararacter-Level%20RNN%2C%20Exercise.ipynb)

	* [Attention Basics](https://github.com/Brandon-HY-Lin/CVND_Exercises/blob/master/2_6_Attention/1_1.%20Attention%20Basics.ipynb)


* Part 3: Object Tracking and Localization
	* Introduction to Motion
		* [Optical Flow and Motion Vectors](https://github.com/Brandon-HY-Lin/udacity_cvnd_optical_flow/blob/master/Optical%20Flow.ipynb)

	* Robot Localization
		* [1D Robot World](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_2_Robot_Localization/1_1.%201D%20Robot%20World%2C%20exercise.ipynb)

		* [Probability After Sense](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_2_Robot_Localization/2_1.%20Probability%20After%20Sense%2C%20exercise.ipynb)

		* [Sense Function](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_2_Robot_Localization/3_1.%20Sense%20Function%2C%20exercise.ipynb)

		* [Normalized Sense Function](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_2_Robot_Localization/4_1.%20Normalized%20Sense%20Function%2C%20exercise.ipynb)

		* [Multiple Measurements](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_2_Robot_Localization/5_1.%20Multiple%20Measurements%2C%20exercise.ipynb)

		* [Move Function](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_2_Robot_Localization/6_1.%20Move%20Function%2C%20exercise.ipynb)

		* [Inexact Move Function](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_2_Robot_Localization/7_1.%20Inexact%20Move%20Function%2C%20exercise.ipynb)

		* [Multiple Moves](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_2_Robot_Localization/8_1.%20Multiple%20Movements%2C%20exercise.ipynb)

		* [Sense and Move Cycle](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_2_Robot_Localization/9_1.%20Sense%20and%20Move%2C%20exercise.ipynb)

	* 2-D Histogram Filter
		* [Two Dimensional Histogram Filter](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/tree/master/4_3_2D_Histogram_Filter)

	* Introduction to Kalman Filter
		* [New Mean and Variance](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_4_Kalman_Filters/2_1.%20New%20Mean%20and%20Variance%2C%20exercise.ipynb)

		* [Predict Function](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_4_Kalman_Filters/3_1.%20Predict%20Function%2C%20exercise.ipynb)

		* [1D Kalman Filter](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_4_Kalman_Filters/4_1.%201D%20Kalman%20Filter%2C%20exercise.ipynb)

	* Representing State and Motion
		* [State and Motion](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/tree/master/4_5_State_and_Motion)

	* Matrices and Transformation of State
		* [Matrices and Transformation of State](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/tree/master/4_6_Matrices_and_Transformation_of_State)

	* Simultaneous Localization and Mapping
		* [Omega and Xi](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/tree/master/4_7_SLAM)

		* [Including Sensor Measurements](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_7_SLAM/2_1.%20Include%20Landmarks%2C%20exercise.ipynb)

		* [Confident Measurements](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/blob/master/4_7_SLAM/3.%20Confident%20Measurements.ipynb)

	* Vehicle Motion and Calculus
		* [Vehicle Motion and Calculus](https://github.com/Brandon-HY-Lin/CVND_Localization_Exercises/tree/master/4_8_Vehicle_Motion_and_Calculus)

		* [Reconstructing Trajectories](https://github.com/Brandon-HY-Lin/udacity_cvnd_Reconstructing_Trajectories)

