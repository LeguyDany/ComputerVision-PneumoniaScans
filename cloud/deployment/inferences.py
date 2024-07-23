import json
import torch
import os

import data_loader
import resnet14
import enums


# TODO
# Add loggers

def model_fn(model_dir):
    device = data_loader.DeviceDataLoader.get_default_device()
    model = data_loader.DeviceDataLoader.to_device(resnet14.ResNet14(3,3), device)

    with open(os.path.join(model_dir, 'resnet14_weights.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def predict_fn(input_data : data_loader.InferenceDataPreparer, model):
    model.eval()
    with torch.no_grad():
        prediction = model(input_data.image_loaded)
        prediction = torch.argmax(prediction, dim=1)
        prediction = enums.EOutput(prediction.item()).name

        return prediction 
        
def input_fn(request_body, request_content_type):
    if request_content_type == 'multipart/form-data':
        image= data_loader.InferenceDataPreparer(
            image_blob=request_body,
            image_size=75
        )
        return image
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def output_fn(prediction_output, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction_output)
    else:
        raise ValueError("Unsupported content type: {}".format(response_content_type))