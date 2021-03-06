import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import preprocess_input


class Featurizer():
    def __init__(self, model_path=None, **model_args):
        '''Create features given a path to a pkl file

        Parameters
        ----------
        model_path: str
            Path to .pkl file to load a tensorflow.keras.Model
            If None a new model is trained given the model_args
        **model_args: dict
            VGG16 model arguments
        
        '''
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        model = tf.keras.models.load_model(model_path)
        # Create the feature extractor part of the network:
        feature_layer_output = model.get_layer('feature_layer').output
        self.feature_extractor = Model(inputs=model.input,
                                       outputs=feature_layer_output)
        self.model = model

    def create_features(self, arrs):
        '''Use tensorflow model to create features of the given input

        The last hidden layer is used for feature creation

        '''
        return self.feature_extractor.predict(preprocess_input(arrs))
