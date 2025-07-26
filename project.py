import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time

detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def run_detector(detector, frame):
    # Convert the frame to a tensor
    input_tensor = tf.convert_to_tensor(frame)
    # Add a batch dimension
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run the model
    start_time = time.time()
    result = detector(input_tensor)
    end_time = time.time()

    # The result contains boxes, scores, classes, and number of detections
    result = {key:value.numpy() for key,value in result.items()}
    
    return result, end_time - start_time

def draw_boxes(frame, result, confidence_threshold=0.5):
    boxes = result['detection_boxes'][0]
    scores = result['detection_scores'][0]
    classes = result['detection_classes'][0].astype(np.int64)
    
    height, width, _ = frame.shape

    for i in range(boxes.shape[0]):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            
            # Scale the bounding box coordinates to the frame size
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            
            # Get the class name
            class_name = COCO_CLASSES[classes[i] - 1] # COCO labels are 1-indexed

            # Draw the bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Draw the label
            label = f"{class_name}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return frame

def main():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the frame
        result, inference_time = run_detector(detector, frame)

        # Draw the bounding boxes on the frame
        frame_with_boxes = draw_boxes(frame.copy(), result)

        # Display the inference time on the frame
        cv2.putText(frame_with_boxes, f"Inference time: {inference_time:.3f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-Time Object Detection', frame_with_boxes)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()