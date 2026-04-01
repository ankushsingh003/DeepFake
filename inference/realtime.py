import cv2
import torch
import numpy as np
from models.detector import DeepfakeDetector
from models.gradcam import GradCAM

def run_realtime_inference():
    """
    Live webcam inference with heatmap toggle.
    """
    model = DeepfakeDetector()
    model.eval()
    
    # Grad-CAM instance for heatmap visualization
    # gradcam = GradCAM(model, model.vit.blocks[-1])
    
    cap = cv2.VideoCapture(0)
    show_heatmap = False
    
    print("Press 'H' to toggle heatmap, 'Q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess frame
        # ...
        
        # Inference
        # output = model(input_tensor)
        
        # visualization = frame.copy()
        # heatmap = gradcam(input_tensor) if show_heatmap else None
        
        # Overlay heatmap if enabled
        # ...
        
        cv2.imshow("Deepfake Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('h'):
            show_heatmap = not show_heatmap
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_inference()
