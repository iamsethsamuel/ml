from matplotlib import cm
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rc('figure', autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=18, titlepad=10)
plt.rc("image", cmap="magma")


img_path = "/Users/sethsamuel/.cache/kagglehub/datasets/ryanholbrook/car-or-truck/versions/1/valid/Car/05123.jpeg"
img = tf.io.read_file(img_path)
img = tf.io.decode_jpeg(img)

kernel_2d = tf.constant([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
], dtype=tf.float32) # Define a kernel for edge detection


# apply the filter
img = tf.image.convert_image_dtype(img, dtype=tf.float32)
img = tf.expand_dims(img, axis=0)  # Add batch dimension

channels = img.shape[-1]

kernel = tf.reshape(kernel_2d, [*kernel_2d.shape, 1, 1])
kernel = tf.expand_dims([kernel_2d] * channels, axis=-1)            # shape: (3, 3, 3, 1)

filtered_img = tf.nn.conv2d(
    input=img,
    filters=kernel,
    strides=1,
    padding="SAME"
)

plt.figure(figsize=(6, 6))

plt.axis("off")
plt.imshow(tf.squeeze(filtered_img), cmap="gray")


# Detection

detected_image = tf.nn.relu(filtered_img)

plt.figure(figsize=(6, 6))

plt.axis("off")
plt.imshow(tf.squeeze(detected_image))
plt.show()
