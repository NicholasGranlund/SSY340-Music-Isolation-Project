import torch

from model import UnetModel


def test_model_forward():
    model = UnetModel()
    device = torch.device('cpu')
#    model.transfer_weights('unet_weight_transfered.pth', device)
    input_tensor = torch.rand(2, 1, 256, 256)
    output_tensor = model.forward(input_tensor)
    print(output_tensor.shape)
    assert output_tensor.shape == (2, 1, 256, 256)


if __name__ == '__main__':
    test_model_forward()