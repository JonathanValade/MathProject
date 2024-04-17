# jo c une bitch
from inference_sdk import InferenceHTTPClient
# import supervision to visualize our results
import supervision as sv
# import cv2 to helo load our image
import cv2


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="I3OqXk01rMiMxjUj1zoj"
)

image2 = "geometry-dash.jpg"
image = cv2.imread(image2)

result = CLIENT.infer(image2, model_id="geometry-dash-ai-detection/4")

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(result)

# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image                                                                           
sv.plot_image(annotated_image)