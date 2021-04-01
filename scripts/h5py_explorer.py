import h5py

for split in ['train', 'val', 'test']:

        with h5py.File('data/' + split + '_cleaned_reduced.h5', "r") as h5_file:

            for k,v in h5_file.items():
                print(k, v)