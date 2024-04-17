from roboflow import Roboflow
rf = Roboflow(api_key="I3OqXk01rMiMxjUj1zoj")
project = rf.workspace().project("projet-math-geometry-dash")
model = project.version(2).model

image_file = 'media/geometry-dash.jpg'

#infer on a local image
print(model.predict(image_file , confidence=40, overlap=30).json())

#visualize your prediction
model.predict(image_file, confidence=40, overlap=30).save("prediction.jpg")