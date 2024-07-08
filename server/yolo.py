import numpy as np
import onnxruntime as rt


def model(img):
    sess = rt.InferenceSession("onnx/pdti_and_unirid.onnx")
    img = img.resize((640, 640))
    img_np = np.array(img)
    img_np = img_np.transpose((2, 0, 1))
    img_np = img_np.astype('float32') / 255.
    img_np = np.expand_dims(img_np, axis=0)

    output = sess.run(None, {sess.get_inputs()[0].name: img_np})

    return output
