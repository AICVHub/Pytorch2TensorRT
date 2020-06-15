import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import argparse

def loadEngine2TensorRT(filepath):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

def do_inference(engine, batch_size, input, output_shape):

    context = engine.create_execution_context()
    output = np.empty(output_shape, dtype=np.float32)

    # 分配内存
    d_input = cuda.mem_alloc(1 * input.size * input.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]

    # pycuda操作缓冲区
    stream = cuda.Stream()
    # 将输入数据放入device
    cuda.memcpy_htod_async(d_input, input, stream)

    start = time.time()
    # 执行模型
    context.execute_async(batch_size, bindings, stream.handle, None)
    # 将预测结果从从缓冲区取出
    cuda.memcpy_dtoh_async(output, d_output, stream)
    end = time.time()

    # 线程同步
    stream.synchronize()

    #
    print("\nTensorRT {} test:".format(engine_path.split('/')[-1].split('.')[0]))
    print("output:", output)
    print("time cost:", end - start)

def get_shape(engine):
    for binding in engine:
        if engine.binding_is_input(binding):
            input_shape = engine.get_binding_shape(binding)
        else:
            output_shape = engine.get_binding_shape(binding)
    return input_shape, output_shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "TensorRT do inference")
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size')
    parser.add_argument("--img_path", type=str, default='test_image/1.jpg', help='cache_file')
    parser.add_argument("--engine_file_path", type=str, default='my_files/test.engine', help='engine_file_path')
    args = parser.parse_args()

    engine_path = args.engine_file_path
    engine = loadEngine2TensorRT(engine_path)
    img = Image.open(args.img_path)
    input_shape, output_shape = get_shape(engine)
    transform = transforms.Compose([
        transforms.Resize([input_shape[1], input_shape[2]]),  # [h,w]
        transforms.ToTensor()
        ])
    img = transform(img).unsqueeze(0)
    img = img.numpy()

    do_inference(engine, args.batch_size, img, output_shape)
