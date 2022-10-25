from unittest import TestCase

import numpy as np

from fedl import onnx_utils
from sklearn.preprocessing import MinMaxScaler
import onnxruntime as rt
import torch
import torch.nn as nn


# Set up commonly used variables.  We set the seed so that the inputs are predictable, but scan a good range.
np.random.seed(seed=732)
mins = np.array([1, 2])
maxs = np.array([2, 4])
x = np.random.random(size=(5, 2)).astype(np.float32)
net = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4, 2),
    nn.Sigmoid(),
    nn.Dropout(),
    nn.Linear(2, 2)
)
net.eval()


def rt_predict(model, x_in):
    session = rt.InferenceSession(model.SerializeToString())
    inputs = {session.get_inputs()[0].name: x_in}
    result = session.run(None, inputs)[0]
    return result


class TestOnnxUtils(TestCase):

    def test_get_inverse_min_max_scaler(self):
        scaler = MinMaxScaler()
        scaler.fit(np.array([mins, maxs]))
        exp = scaler.inverse_transform(x)

        model = onnx_utils.get_inverse_min_max_scaler(mins=mins, maxs=maxs)
        result = rt_predict(model, x)

        np.testing.assert_allclose(exp, result, rtol=1e-03, atol=1e-05)

    def test_get_min_max_scaler(self):
        scaler = MinMaxScaler()
        scaler.fit(np.array([mins, maxs]))
        exp = scaler.transform(x)

        model = onnx_utils.get_min_max_scaler(mins=mins, maxs=maxs)
        result = rt_predict(model, x)

        np.testing.assert_allclose(exp, result, rtol=1e-03, atol=1e-05)

    def test_convert_sklearn_min_max(self):
        scaler = MinMaxScaler()
        scaler.fit(np.array([mins, maxs]))
        exp = scaler.transform(x)

        model = onnx_utils.convert_sklearn_min_max(scaler, inverse=False)
        result = rt_predict(model, x)

        np.testing.assert_allclose(exp, result, rtol=1e-03, atol=1e-05)

    def test_convert_sklearn_min_max_inverse(self):
        scaler = MinMaxScaler()
        scaler.fit(np.array([mins, maxs]))
        exp = scaler.inverse_transform(x)

        model = onnx_utils.convert_sklearn_min_max(scaler, inverse=True)
        result = rt_predict(model, x)

        np.testing.assert_allclose(exp, result, rtol=1e-03, atol=1e-05)

    def test_convert_to_onnx_only_model(self):
        xt = torch.tensor(x, requires_grad=False)
        exp = net(xt).detach().numpy()

        model = onnx_utils.convert_to_onnx(torch_model=net, x=x, input_name='inp', output_name='out',
                                           input_transform=None, target_transform=None)
        result = rt_predict(model, x)

        np.testing.assert_allclose(exp, result, rtol=1e-03, atol=1e-05)

    def test_convert_to_onnx_preprocess(self):
        scaler = MinMaxScaler()
        scaler.fit([mins, maxs])

        xs = scaler.transform(x)
        xt = torch.tensor(xs, requires_grad=False)
        exp = net(xt).detach().numpy()

        model = onnx_utils.convert_to_onnx(torch_model=net, x=x, input_name='inp', output_name='out',
                                           input_transform=scaler, target_transform=None)
        result = rt_predict(model, x)

        np.testing.assert_allclose(exp, result, rtol=1e-03, atol=1e-05)

    def test_convert_to_onnx_postprocess(self):
        scaler = MinMaxScaler()
        scaler.fit([mins, maxs])

        xt = torch.tensor(x, requires_grad=False)
        exp = scaler.inverse_transform(net(xt).detach().numpy())

        model = onnx_utils.convert_to_onnx(torch_model=net, x=x, input_name='inp', output_name='out',
                                           input_transform=None, target_transform=scaler)
        result = rt_predict(model, x)

        np.testing.assert_allclose(exp, result, rtol=1e-03, atol=1e-05)

    def test_convert_to_onnx_all(self):
        i_scaler = MinMaxScaler()
        i_scaler.fit([mins, maxs])
        o_scaler = MinMaxScaler()
        o_scaler.fit([mins, maxs])

        xs = i_scaler.transform(x)
        xt = torch.tensor(xs, requires_grad=False)
        exp = o_scaler.inverse_transform(net(xt).detach().numpy())

        model = onnx_utils.convert_to_onnx(torch_model=net, x=x, input_name='inp', output_name='out',
                                           input_transform=i_scaler, target_transform=o_scaler)
        result = rt_predict(model, x)

        np.testing.assert_allclose(exp, result, rtol=1e-03, atol=1e-05)

    def test_convert_to_onnx_devices(self):
        if torch.has_cuda:
            device = torch.device("cuda:0")
            i_scaler = MinMaxScaler()
            i_scaler.fit([mins, maxs])
            o_scaler = MinMaxScaler()
            o_scaler.fit([mins, maxs])

            xs = i_scaler.transform(x)
            xt = torch.tensor(xs, requires_grad=False).to(device)
            exp = o_scaler.inverse_transform(net(xt).detach().numpy())

            model = onnx_utils.convert_to_onnx(torch_model=net, x=x, input_name='inp', output_name='out',
                                               input_transform=i_scaler, target_transform=o_scaler)
            result = rt_predict(model, x)

            np.testing.assert_allclose(exp, result, rtol=1e-03, atol=1e-05)
        else:
            print('CUDA not available for testing tensor and model on different device case')
