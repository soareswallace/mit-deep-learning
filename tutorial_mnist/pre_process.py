def preprocess_images(imgs):
    sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
    assert sample_img.shape == (28, 28), sample_img.shape
    return imgs/255