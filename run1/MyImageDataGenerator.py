from keras.preprocessing.image import ImageDataGenerator


# overload the current generator standardize function to fix some inconsistencies
class MyImageDataGenerator (ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x /= 255.
            x -= 0.5
            x *= 2.
        return x
