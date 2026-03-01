from torch.utils.data import ConcatDataset

from dataset.gen1 import Gen1
from dataset.ncaltech101 import NCaltech101


def get_dataset(
        dataset_name: str,
        window_size: float = 0.1,
        do_validation: bool = True,
    ):
    """
    Returns instances of train, validation and test datasets of the supported datasets, specified by name.
    """
    if dataset_name == 'Gen1':
        # Data: events with format <t, x, y, p>
        # Label: bboxes with format <x_center, y_center, w, h, class_id>
        train_dataset = Gen1('data/Gen1', split='train',
                            window_size=window_size, valid_idx_path='data/Gen1/idx_train_01.json')
        test_dataset = Gen1('data/Gen1', split='test', 
                            window_size=window_size, valid_idx_path='data/Gen1/idx_test_01.json')
        if do_validation:
            val_dataset = Gen1('data/Gen1', split='val',
                                window_size=window_size, valid_idx_path='data/Gen1/idx_val_01.json')
        else:
            val_dataset = Gen1('data/Gen1', split='val',
                                window_size=window_size, valid_idx_path='data/Gen1/idx_val_01.json')
            train_dataset = ConcatDataset([train_dataset, val_dataset])
            val_dataset = test_dataset

    elif dataset_name == 'NCaltech101':
        train_dataset = NCaltech101('data/NCaltech101', split='train',
                                    window_size=window_size)
        test_dataset = NCaltech101('data/NCaltech101', split='test',
                                   window_size=window_size)
        if do_validation:
            val_dataset = NCaltech101('data/NCaltech101', split='val',
                                      window_size=window_size)
        else:
            val_dataset = NCaltech101('data/NCaltech101', split='val',
                                      window_size=window_size, aug_on_validation=True)
            train_dataset = ConcatDataset([train_dataset, val_dataset])
            val_dataset = test_dataset
    
    else:
        raise ValueError(f'Dataset {dataset_name} is not supported.')
    return train_dataset, val_dataset, test_dataset
