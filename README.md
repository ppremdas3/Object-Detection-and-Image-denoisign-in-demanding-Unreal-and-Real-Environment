# Object-Detection-and-Image-denoisign-in-demanding-Unreal-and-Real-Environment

## Overview

Welcome to the Image Denoising and Enhancement Project! This project explores and compares the effectiveness of three key methods—Adaptive Gamma Correction, Anisotropic Diffusion, and the MAXIM ML model—in mitigating various types of image noise and enhancing overall image quality. The objective is to provide valuable insights for practitioners and researchers working in fields such as medical imaging, computer vision, and remote sensing.

## Motivation


This project is motivated by the paramount importance of achieving high-quality images in various applications, including medical imaging, computer vision, and remote sensing. The ubiquitous presence of noise in digital images poses a significant challenge, necessitating robust denoising and enhancement methods. In this context, our focus on comparing three distinct approaches—Adaptive Gamma Correction, Anisotropic Diffusion, and the MAXIM ML model—addresses the critical need for effective strategies in handling diverse noise types. The outcomes of this project hold promise for improving image quality across industries, enhancing the accuracy of medical diagnoses, facilitating precise computer vision tasks, and ensuring reliable analysis of satellite imagery. By contributing insights into the strengths and limitations of these methods, this research not only aids practitioners in method selection but also contributes to the broader advancement of denoising and enhancement techniques, thereby addressing a fundamental challenge in image processing.


## Project Structure

- `code/`: Contains the implementation of the denoising and enhancement methods in Python.
- `figures/`: Includes the images of the results generated for the report.
- `results/`: Stores the IQA comparision table and excel files with IQA for each dataset including all types of noises and all noise levels
- `docs/`: Report and poster of the project.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/image-denoising-project.git

2. **Download the following models:**

   MAXIM MLP image denoising/enhancement: Dowload the enhancement, denoising, deblurring, deraining and dehazing from https://www.kaggle.com/models/kaggle/maxim/frameworks/TensorFlow2/variations/s-3-deblurring-realblur-r/versions/1
   Object deetction: Download the YOLOv8x from https://huggingface.co/yaroslavski88/Yolov8_Object_detection_v1.0/tree/main/weights/detection
                     Download traffic sign model form https://huggingface.co/JakobJFL/yolov8-dk-Traffic-Signs/tree/main

## Running Experiment

1. Run the jupyter notebooks in 'eval/' folder to execute the code.
2. Check 'results/' directory for output table

   
