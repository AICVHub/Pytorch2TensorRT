import argparse
from trt_convertor import ONNX2TRT
from myCalibrator import CenterNetEntropyCalibrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Pytorch2TensorRT args")
    parser.add_argument("--batch_size", type=int, default=32, help='batch_size')
    parser.add_argument("--channel", type=int, default=3, help='input channel')
    parser.add_argument("--height", type=int, default=288, help='input height')
    parser.add_argument("--width", type=int, default=512, help='input width')
    parser.add_argument("--cache_file", type=str, default='', help='cache_file')
    parser.add_argument("--mode", type=str, default='', help='fp32, fp16 or int8')
    parser.add_argument("--onnx_file_path", type=str, default='', help='onnx_file_path')
    parser.add_argument("--engine_file_path", type=str, default='', help='engine_file_path')
    args = parser.parse_args()
    print(args)
    if args.mode.lower() == 'int8':
        # Note that: if use int8 mode, you should prepare a calibrate dataset and create a Calibrator class.
        # In Calibrator class, you should override 'get_batch_size, get_batch',
        # 'read_calibration_cache', 'write_calibration_cache'.
        # You can reference implementation of CenterNetEntropyCalibrator.
        calib = CenterNetEntropyCalibrator()
    else:
        calib = None
    ONNX2TRT(args, calib=calib)

