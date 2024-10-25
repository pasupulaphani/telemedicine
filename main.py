import aivm_client as aic
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


# List all supported models
available_models = aic.get_supported_models()
print(available_models)

trans = transforms.Compose(
    [
        transforms.ToTensor(),                # Transform to tensor
        transforms.Resize((28, 28)),          # Resize to (28 x 28)
        transforms.Normalize((0.5,), (1.0,)), # Normalize the image
    ]
)
# Load the dataset
dataset = dset.MNIST(
    root="/tmp/mnist", train=True, transform=trans, download=True
)

# Get entry #20 of the dataset
inputs, _ =  dataset[20]
inputs = inputs.reshape(1, 1, 28, 28)

encrypted_input = aic.LeNet5Cryptensor(inputs)


result = aic.get_prediction(encrypted_input, "LeNet5MNIST")

print(result)
