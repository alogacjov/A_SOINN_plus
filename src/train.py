import argparse
from models import model
from utils import config
from utils import data_loader
from utils import logger


def train(cfg, dataset_path=None):
    '''Train either GDM or ASOINN+

    Parameters
    ----------
    cfg: utils.config.Config
        model dependent config
    dataset_path: str
        if not given, dataset specified in config used

    '''
    # Dataset
    ds_path = cfg.DS_PATH if dataset_path is None else dataset_path
    dl = data_loader.DataLoader(
        ds_path=ds_path,
        fe_path=cfg.FE_PATH,
        objects=cfg.train_labels,
        sessions=cfg.sessions,
        category_indices=cfg.category_indices,
        instance_indices=cfg.instance_indices,
        random_seed=cfg.SEED
    )
    test_dl = data_loader.DataLoader(
        ds_path=ds_path,
        fe_path=cfg.FE_PATH,
        objects=cfg.test_labels,
        sessions=cfg.sessions,
        category_indices=cfg.category_indices,
        instance_indices=cfg.instance_indices,
        random_seed=cfg.SEED,
        keep_prev_batch=True
    )
    # Logging
    _logger = logger.Logger(
        'logs/',
        'average_accuracy',
        'average_forgetting'
    )
    # Model
    m = model.Model(cfg.ALGORITHM)
    m.train(dataset=dl, test_dataset=test_dl, logger=_logger,
            args=cfg.ALGORITHM_ARGS)
    _logger.write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start LL training.')
    parser.add_argument('-c', '--config_path', required=True, type=str,
                        help='model path with config.yml file')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default=None)
    args = parser.parse_args()
    cfg = config.Config(args.config_path)
    ds_path = args.dataset_path
    train(cfg, ds_path)
