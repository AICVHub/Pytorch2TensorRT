# -*- coding: utf-8 -*-

import tensorrt as trt

def ONNX2TRT(args, calib=None):
    ''' convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
    :return: trt engine
    '''

    assert args.mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"

    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # TRT7中的onnx解析器的network，需要指定EXPLICIT_BATCH
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:

        builder.max_batch_size = args.batch_size
        builder.max_workspace_size = 1 << 30
        if args.mode.lower() == 'int8':
            assert (builder.platform_has_fast_int8 == True), "not support int8"
            builder.int8_mode = True
            builder.int8_calibrator = calib
        elif args.mode.lower() == 'fp16':
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            builder.fp16_mode = True

        print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
        with open(args.onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for e in range(parser.num_errors):
                    print(parser.get_error(e))
                raise TypeError("Parser parse failed.")

        print('Completed parsing of ONNX file')

        print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Created engine success! ")

        # 保存计划文件
        print('Saving TRT engine file to path {}...'.format(args.engine_file_path))
        with open(args.engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print('Engine file has already saved to {}!'.format(args.engine_file_path))
        return engine


def loadEngine2TensorRT(filepath):
    '''
    通过加载计划文件，构建TensorRT运行引擎
    '''
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine
