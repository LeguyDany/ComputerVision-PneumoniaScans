import os
import inferences

def main():
    model = inferences.model_fn(model_dir="cloud/deployment")

    file = os.listdir(os.path.join(os.getcwd(), "dataset", "TRAIN"))[112]
    print(file)
    image_path = os.path.join(os.getcwd(), "dataset", "TRAIN", file)
    # image_path = os.path.join(os.getcwd(), "..", "maxresdefault.jpg")

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    image = inferences.input_fn(image_bytes, 'multipart/form-data')
    pred = inferences.predict_fn(image, model)
    output = inferences.output_fn(pred, 'application/json')
    print(f"Prediction: {output}")

if __name__ == '__main__':
    main()