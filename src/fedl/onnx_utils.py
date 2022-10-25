"""This module contains simple pre- and post-processing code for packaging entire pipelines in ONNX format.

Here is an example using a the min_max_scalers.

import numpy as np
import onnx
import onnxruntime as rt

mins = np.array([[1, 2, 3, 4,]]).astype(np.float32)
maxs = np.array([[2, 4, 6, 8,]]).astype(np.float32)

# Setup the pre and post processing models
pre = get_min_max_scaler(mins=mins, maxs=maxs)
post = get_inverse_min_max_scaler(mins=mins, maxs=maxs)

# Compose the models together.  Here you would typically supply your own model in-between pre and post.  io_map matches
# the output of an earlier model with the input of its successor model
model = onnx.compose.merge_models(pre, post, io_map=[('mms/xs', 'imms/x')])
sess = rt.InferenceSession(model.SerializeToString())

x = np.array([[1.5, 3, 4.5, 6]]).astype(np.float32)
sess.run(['imms/xb'], {'mms/x': x})

"""
import io
from typing import Optional, List

import onnx
import onnx.helper as helper
import onnx.checker as checker
import sklearn.preprocessing
import torch
from onnx import TensorProto as tp
import numpy as np
import onnxruntime as rt
from sklearn.preprocessing import MinMaxScaler
from torch import nn


OPSET_VERSION = 16


def convert_sklearn_min_max(scaler, inverse=False):
    """Use the protected attributes to convert a fitted sklearn MinMaxScaler to custom onnx value.

    sklearn does provide it's own implementation, but it does not provide the inverse.
    """
    if inverse:
        return get_inverse_min_max_scaler(mins=scaler.data_min_, maxs=scaler.data_max_)
    else:
        return get_min_max_scaler(mins=scaler.data_min_, maxs=scaler.data_max_)


def get_min_max_scaler(mins: np.ndarray, maxs: np.ndarray, prefix: str = 'mms/') -> onnx.ModelProto:
    """Perform Min-Max scaling natively in ONNX.

    This does the following conversion: xhat = (x - xmin) / (xmax - xmin)
    """

    shape = ('batch_size', *mins.shape)

    # The required constants:
    # The minimum value of each feature
    c1 = helper.make_node('Constant', inputs=[], outputs=['c1'], name='c1-node',
                     value=helper.make_tensor(name='c1v', data_type=tp.FLOAT, dims=mins.shape,
                                         vals=mins.flatten()))

    # The range of each feature
    c2 = helper.make_node('Constant', inputs=[], outputs=['c2'], name='c2-node',
                     value=helper.make_tensor(name='c2v', data_type=tp.FLOAT, dims=mins.shape,
                                         vals=(maxs-mins).flatten()))


    # The functional nodes:
    # Unbias X by its min value
    n1 = helper.make_node('Sub', inputs=['x', 'c1'], outputs=['xub'], name='n1')
    # Scale the unbiased X by its observed ranged
    n2 = helper.make_node('Div', inputs=['xub', 'c2'], outputs=['xs'], name='n2')

    # Create the graph
    g1 = helper.make_graph([c1, n1, c2, n2], 'min-max-scaler',
                      [helper.make_tensor_value_info('x', tp.FLOAT, shape)],
                      [helper.make_tensor_value_info('xs', tp.FLOAT, shape)])

    op = onnx.OperatorSetIdProto()
    op.version = OPSET_VERSION
    m1 = helper.make_model(g1, producer_name='adamc', opset_imports=[op])
    # del m1.opset_import[:]
    # opset = m1.opset_import.add()
    # opset.domain = ''
    # opset.version = OPSET_VERSION
    onnx.compose.add_prefix(model=m1, prefix=prefix, inplace=True)
    checker.check_model(m1)

    return m1


def get_inverse_min_max_scaler(mins: np.ndarray, maxs: np.ndarray, prefix: str = 'imms/') -> onnx.ModelProto:
    """Perform inverse min-max scaling natively in ONNX.

    This does the following conversion:  xhat =  x * (xmax - xmin) + xmin.
    """

    shape = ('batch_size', *mins.shape)
    # The required constants:
    # The range of each feature
    c1 = helper.make_node('Constant', inputs=[], outputs=['c1'], name='c1-node',
                          value=helper.make_tensor(name='c1v', data_type=tp.FLOAT, dims=mins.shape,
                                                   vals=(maxs - mins).flatten()))

    # The minimum value of each feature
    c2 = helper.make_node('Constant', inputs=[], outputs=['c2'], name='c2-node',
                          value=helper.make_tensor(name='c2v', data_type=tp.FLOAT, dims=mins.shape,
                                                   vals=mins.flatten()))

    # The functional nodes:
    # Scale the input by the observed feature ranges
    n1 = helper.make_node('Mul', inputs=['x', 'c1'], outputs=['xs'], name='n1',)
    # Rebias those features by adding back the observerd min values
    n2 = helper.make_node('Add', inputs=['xs', 'c2'], outputs=['xb'], name='n2')

    # Create the graph
    g1 = helper.make_graph([c1, n1, c2, n2], 'inv-min-max-scaler',
                           [helper.make_tensor_value_info('x', tp.FLOAT, shape)],
                           [helper.make_tensor_value_info('xb', tp.FLOAT, shape)])

    op = onnx.OperatorSetIdProto()
    op.version = OPSET_VERSION
    m1 = helper.make_model(g1, producer_name='adamc', opset_imports=[op])
    # del m1.opset_import[:]
    # opset = m1.opset_import.add()
    # opset.domain = ''
    # opset.version = OPSET_VERSION
    onnx.compose.add_prefix(model=m1, prefix=prefix, inplace=True)
    checker.check_model(m1)

    return m1


def convert_to_onnx(torch_model: nn.Module, x: np.ndarray, input_name: str, output_name: str,
                    input_transform: Optional[MinMaxScaler] = None, target_transform: Optional[MinMaxScaler] = None):
    """Convert a model to ONNX format.  This assumes only a single output and single input.
    Args:
        torch_model: A pytorch model
        x: An untransformed input to model.  requires_grad should be True for this
        input_name: The name of the input to the model
        output_name: The name of the outputs of the model
        input_transform: An object that handles transforming the inputs for use in the model (e.g., sklearn MinMaxScaler)
        target_transform: An object that handles transforming the outputs of the model for human interpretation (e.g., sklearn MinMaxScaler)
    """

    if input_transform is not None and type(input_transform).__name__ != 'MinMaxScaler':
        raise TypeError("input_transform must be of type MinMaxScaler")
    if target_transform is not None and type(target_transform).__name__ != 'MinMaxScaler':
        raise TypeError("output_transform must be of type MinMaxScaler")

    # What device is the model using
    device = next(torch_model.parameters()).device

    # Transform the input if specified
    torch_in = torch.tensor(x, requires_grad=False).to(device)
    preprocess = None
    if input_transform is not None:
        torch_in = torch.tensor(input_transform.transform(x), requires_grad=False).to(device)
        preprocess = convert_sklearn_min_max(input_transform, inverse=False)

    # Convert the model to ONNX format.  We need eval mode so we can compare the results later (no dropout, etc.)
    torch_model.eval()
    onnx_model_save = io.BytesIO()
    torch.onnx.export(torch_model, torch_in, onnx_model_save, export_params=True, do_constant_folding=True,
                      input_names=[input_name], output_names=[output_name], opset_version=OPSET_VERSION,
                      dynamic_axes={input_name: {0: 'batch_size'}, output_name: {0: 'batch_size'}})

    # Check that the model is well-formed
    onnx_model = onnx.load_model_from_string(onnx_model_save.getvalue())
    onnx.checker.check_model(onnx_model)

    # Make a prediction and convert the output if specified
    torch_out = torch_model(torch_in).detach().cpu().numpy()
    postprocess = None
    if target_transform is not None:
        torch_out = target_transform.inverse_transform(torch_out)
        postprocess = convert_sklearn_min_max(target_transform, inverse=True)

    # Chain the models together if needed.  Prefix the model in case it's chained, then rename the final inputs and
    # outputs.
    onnx.compose.add_prefix(model=onnx_model, prefix="model/", inplace=True)
    if preprocess is not None:
        onnx_model = onnx.compose.merge_models(preprocess, onnx_model, io_map=[('mms/xs', f"model/{input_name}")])
    if postprocess is not None:
        onnx_model = onnx.compose.merge_models(onnx_model, postprocess, io_map=[(f"model/{output_name}", 'imms/x')])

    rename_model_inputs(model=onnx_model,
                        old_names=[onnx_model.graph.input[i].name for i in range(len(onnx_model.graph.input))],
                        new_names=[input_name])
    rename_model_outputs(model=onnx_model,
                         old_names=[onnx_model.graph.output[i].name for i in range(len(onnx_model.graph.input))],
                         new_names=[output_name])

    # Check that the ONNX model makes the same prediction as the pytorch model
    ort_session = rt.InferenceSession(onnx_model.SerializeToString())
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-03, atol=1e-05)

    return onnx_model


def rename_model_inputs(model, new_names: List[str], old_names: List[str]):
    """Rename the inputs of the model and their references throughout the graph."""
    if len(new_names) != len(old_names):
        raise ValueError("Must supply the same number of old names as new")

    for (new, old) in zip(new_names, old_names):
        for i in range(len(model.graph.node)):
            for j in range(len(model.graph.node[i].input)):
                if model.graph.node[i].input[j] == old:
                    model.graph.node[i].input[j] = new

        for i in range(len(model.graph.input)):
            if model.graph.input[i].name == old:
                model.graph.input[i].name = new


def rename_model_outputs(model, new_names: List[str], old_names: List[str]):
    """Rename the outputs of the model and their references throughout the graph."""
    if len(new_names) != len(old_names):
        raise ValueError("Must supply the same number of old names as new")

    for (new, old) in zip(new_names, old_names):
        for i in range(len(model.graph.node)):
            for j in range(len(model.graph.node[i].output)):
                if model.graph.node[i].output[j] == old:
                    model.graph.node[i].output[j] = new

        for i in range(len(model.graph.output)):
            if model.graph.output[i].name == old:
                model.graph.output[i].name = new
