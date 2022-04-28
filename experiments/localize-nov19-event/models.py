from typing import List
import torch.nn as nn


class MLPRelu(nn.Module):
    """Class for a basic multilayer perceptron with ReLU and dropout."""

    def __init__(self, n_inputs, n_outputs, layers: List[int], dropout_input_p: float = 0.2, dropout_p=0.5) -> None:
        if len(layers) < 1:
            raise RuntimeError("Network must have at least one hidden")
        super(MLPRelu, self).__init__()

        model_dims = [n_inputs, *layers, n_outputs]
        layer_list = []
        for i in range(len(model_dims)):
            # Handle the last layer differently since we just want the prediction
            layer_list.append(nn.Linear(model_dims[i], model_dims[i + 1]))
            if i == (len(model_dims) - 2):
                break
            else:
                # Add dropout and relu if it's not input or output layer
                layer_list.append(nn.ReLU())
                if i == 0:
                    layer_list.append(nn.Dropout(dropout_input_p))
                else:
                    layer_list.append(nn.Dropout(dropout_p))
        self.dense_net = nn.Sequential(*layer_list)

    def forward(self, samples):
        return self.dense_net(samples)
