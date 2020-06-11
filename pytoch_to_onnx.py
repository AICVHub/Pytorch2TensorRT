# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:35:24 2020

@author: LWS

An example of convert Pytroch model to onnx.
You should import your model and provide input according your model.
"""
import torch

def get_model():
    """ Define your own model and return it
    :return: Your own model
    """
    pass

def get_onnx(model, onnx_save_path, example_tensor):

    example_tensor = example_tensor.cuda()

    _ = torch.onnx.export(model,  # model being run
                                  example_tensor,  # model input (or a tuple for multiple inputs)
                                  onnx_save_path,
                                  verbose=False,  # store the trained parameter weights inside the model file
                                  training=False,
                                  do_constant_folding=True,
                                  input_names=['input'],
                                  output_names=['output']
                                  )

if __name__ == '__main__':

    model = get_model()
    onnx_save_path = "onnx/resnet50_2.onnx"
    example_tensor = torch.randn(1, 3, 288, 512, device='cuda')

    # 导出模型
    get_onnx(model, onnx_save_path, example_tensor)

