from inference_sdk import InferenceHTTPClient
import os
import cv2
import glob
import json
import numpy as np

class ImageInferenceProcessor:
    def __init__(self, api_url, api_key, input_dir, output_dir, model_id):
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_id = model_id
        os.makedirs(self.output_dir, exist_ok=True)
        self.input_images = self._get_input_images()
        self.output_frames = []

    def _get_input_images(self):
        images = glob.glob(os.path.join(self.input_dir, "*.jpg")) + glob.glob(os.path.join(self.input_dir, "*.png"))
        images.sort()
        return images

    def draw_text_with_background(self, image, text, position, font, font_scale, font_thickness, text_color, bg_color):
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x, text_y = position
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 10, text_y + 5), bg_color, -1)
        cv2.putText(image, text, (text_x + 5, text_y), font, font_scale, text_color, font_thickness)

    def process_images(self):
        for img_path in self.input_images:
            print(f"\nProcessing image: {img_path}")
            rgb_frame = cv2.imread(img_path)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            result = self.client.infer(rgb_frame, model_id=self.model_id)
            if isinstance(result, str):
                result = json.loads(result)
            self._process_predictions(rgb_frame, result)
            self._save_processed_image(rgb_frame, img_path)

    def _process_predictions(self, rgb_frame, result):
        if isinstance(result, dict) and 'predictions' in result:
            predictions = result['predictions']
            if predictions:
                top_class = max(predictions, key=lambda k: predictions[k]['confidence'])
                confidence = predictions[top_class]['confidence']
                height, width, _ = rgb_frame.shape
                cv2.rectangle(rgb_frame, (0, 0), (width-1, height-1), (0, 255, 0), 2)
                label = f"{top_class.upper()} {confidence * 100:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 1
                text_color = (255, 255, 255)
                bg_color = (0, 0, 0)
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x = width - text_size[0] - 20
                text_y = 30
                self.draw_text_with_background(rgb_frame, label, (text_x, text_y), font, font_scale, font_thickness, text_color, bg_color)
                print(f"Top classification: {top_class} ({confidence:.4f})")
            else:
                print("No predictions found in the result")
        else:
            print("Unexpected result format:", result)

    def _save_processed_image(self, rgb_frame, img_path):
        output_filename = os.path.join(self.output_dir, os.path.basename(img_path))
        cv2.imwrite(output_filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        print(f"Saved processed image: {output_filename}")
        self.output_frames.append(rgb_frame)

    def create_video(self):
        if not self.output_frames:
            print("No frames to create a video.")
            return
        output_video_path = os.path.join(self.output_dir, "output_video.mp4")
        height, width, _ = self.output_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        total_frames = len(self.output_frames)
        fps = min(30, total_frames / 15)
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        for frame in self.output_frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
        print(f"\nProcessed {len(self.input_images)} images. Output saved to {self.output_dir}")
        print(f"Video created: {output_video_path}")

if __name__ == "__main__":
    processor = ImageInferenceProcessor(
        api_url="https://classify.roboflow.com",
        api_key="YOUR_API_KEY",  # Ensure this is your actual API key
        input_dir=r"E:\roboflow model\input_images",
        output_dir=r"E:\roboflow model\output_images",
        model_id="classification-waste/11"
    )
    processor.process_images()
    processor.create_video()