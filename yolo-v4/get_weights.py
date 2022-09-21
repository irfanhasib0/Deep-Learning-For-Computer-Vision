from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
H = W = 160
model = MobileNetV2(
    input_shape=(H,W,3),
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
)
#model.load_weights(f'model_data/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_{H}.h5')
model.save(f'model_data/mobilenet_v2_1.0_{H}_mod.h5')