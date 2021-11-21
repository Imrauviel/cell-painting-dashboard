from keras.applications.vgg16 import VGG16
from keras.models import Model

model = VGG16()
Vgg16Features = Model(inputs=model.inputs, outputs=model.layers[-2].output)

