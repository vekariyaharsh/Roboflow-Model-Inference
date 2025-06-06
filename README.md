# Roboflow Model Inference

This project showcases an image classification pipeline using a pre-trained model from Roboflow. The primary objective is to automate the process of identifying and classifying objects within images, specifically focusing on waste classification. The project leverages machine learning to process input images, draw bounding boxes around detected objects, and annotate them with class labels and confidence scores. Additionally, it compiles the processed images into a video, providing a visual summary of the classification results.

## Key Features

- **Automated Image Processing**: Efficiently processes batches of images to identify and classify objects.
- **Bounding Box Visualization**: Draws bounding boxes around detected objects, enhancing visual understanding of the model's predictions.
- **Confidence Annotation**: Displays class labels and confidence percentages, offering insights into the model's certainty.
- **Video Compilation**: Converts processed images into a video format, facilitating easy review and presentation of results.
- **Customizable and Extensible**: Designed to be adaptable for various classification tasks beyond waste management, with potential applications in fields like environmental monitoring, inventory management, and more.

## Potential Applications

- **Environmental Monitoring**: Classify and track waste types in environmental studies or waste management facilities.
- **Inventory Management**: Automate the classification of items in warehouses or retail settings.
- **Research and Development**: Serve as a foundation for developing more advanced image classification systems or integrating with other AI-driven solutions.

This project serves as a practical example of how machine learning models can be applied to real-world problems, providing a foundation for further exploration and development in the field of computer vision.

## Project Structure

- `dataset/input_images/`: Directory for input images.
- `dataset/output_images/`: Directory for processed images and video.
- `ImageInference.py`: Main script for processing images.

## Dataset

The dataset used in this project is sourced from the [Classification Waste Dataset](https://universe.roboflow.com/gkhang/classification-waste) on Roboflow Universe. This dataset includes various classes such as biodegradable, cardboard, glass, gloves, masks, medicines, metal, paper, plastic, and syringe.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Harshal1917/Roboflow-Model-Inference.git
   cd roboflow_model_inference
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your input images in the `dataset/input_images/` directory.

4. **Obtain Your API Key**:
   - Visit the [Classification Waste Dataset](https://universe.roboflow.com/gkhang/classification-waste) page on Roboflow Universe.
   - Navigate to the model section and generate your API key.
   - Update the `ImageInference.py` script with your API key:

     ```python
     api_key="YOUR_API_KEY"
     ```

5. **Using on Another Machine or with a Different Model**:
   - If you want to run this project on another laptop or use a different model, update the following in `ImageInference.py`:
   
     ```python
     api_key="YOUR_NEW_API_KEY"
     model_id="your-new-model-id"
     ```

## Usage

Run the script to process images and create a video:
```bash
python ImageInference.py
```

## Results

Processed images and the output video will be saved in the `dataset/output_images/` directory.

### Example Input and Output

Below are examples of input images and their corresponding output images processed by the model. The output images include bounding boxes and labels indicating the detected class and confidence percentage.

#### Input Images

<div style="display: flex; flex-wrap: wrap; gap: 20px;">

  <div style="flex: 1; text-align: center;">
    Gloves Input<br>
    <img src="dataset/input_images/gloves128_jpg.rf.582f0d2219d7b688afc31f0ef87f148b.jpg" alt="Gloves Input" width="300">
  </div>

  <div style="flex: 1; text-align: center;">
    Masks Input<br>
    <img src="dataset/input_images/masks339_jpg.rf.66ef8d59912da373d35ea7b556537034.jpg" alt="Masks Input" width="300">
  </div>

</div>

#### Output Images
<div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px;">

  <div style="flex: 1; text-align: center;">
    Gloves Output<br>
    <img src="dataset/output_images/gloves128_jpg.rf.582f0d2219d7b688afc31f0ef87f148b.jpg" alt="Gloves Output" width="300">
  </div>

  <div style="flex: 1; text-align: center;">
    Masks Output<br>
    <img src="dataset/output_images/masks339_jpg.rf.66ef8d59912da373d35ea7b556537034.jpg" alt="Masks Output" width="300">
  </div>

</div>

## Acknowledgments

The dataset used in this project is provided by [GKHANG](https://universe.roboflow.com/gkhang/classification-waste) on Roboflow Universe. If you use this dataset in a research paper, please cite it using the following BibTeX:

```bibtex
@misc{
    classification-waste_dataset,
    title = { Classification waste Dataset },
    type = { Open Source Dataset },
    author = { GKHANG },
    howpublished = { \url{ https://universe.roboflow.com/gkhang/classification-waste } },
    url = { https://universe.roboflow.com/gkhang/classification-waste },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2024 },
    month = { oct },
    note = { visited on 2024-10-10 },
}
```
