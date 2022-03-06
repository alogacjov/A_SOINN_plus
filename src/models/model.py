

class Model():
    def __init__(self, architecture_name):
        '''Base model class

        Parameters
        ----------
        architecture_name: str
            either A_SOINN_plus or GDM

        '''
        self.architecture_name = architecture_name

    def train(self, dataset, args, test_dataset=None, logger=None):
        '''Train defined model

        Parameters
        ----------
        dataset: data_loader.DataLoader
        test_dataset: data_loader.DataLoader
        args: dict
            Model parameters
        logger: utils.logger.Logger

        '''
        if self.architecture_name == 'A_SOINN_plus':
            from models import a_soinn_plus
            self.model = a_soinn_plus.ASOINNPlus()
            self.model.train(
                dataset=dataset,
                test_dataset=test_dataset,
                num_context=args['NUM_CONTEXT'],
                learning_rates=[args['LR_BMU'], args['LR_sBMU']],
                only_each_nth=args['SKIP_FRAMES'],
                epochs=args['EPOCHS'],
                creation_constraint=args['CREATION_CONSTRAINT'],
                adaptation_constraint=args['ADAPTATION_CONSTRAINT'],
                input_dimension=args['INPUT_DIMENSION'],
                logger=logger
            )
        elif self.architecture_name == 'GDM':
            from models import gdm
            self.model = gdm.GDM(
                replay=args['REPLAY'],
                semantic=args['SEMANTIC']
            )
            self.model.train(
                dataset=dataset,
                test_dataset=test_dataset,
                max_age=args['MAX_AGE'],
                num_context=args['NUM_CONTEXT'],
                learning_rates=[args['LR_BMU'], args['LR_sBMU']],
                a_threshold=[args['A_THRESHOLD_GEM'],
                             args['A_THRESHOLD_GSM']],
                only_each_nth=args['SKIP_FRAMES'],
                epochs=args['EPOCHS'],
                input_dimension=args['INPUT_DIMENSION'],
                logger=logger
            )
        else:
            raise NotImplementedError(f'Unknown: {self.architecture_name}')
