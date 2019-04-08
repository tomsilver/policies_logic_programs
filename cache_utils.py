from scipy.sparse import csr_matrix, save_npz, load_npz

import glob
import pickle
import os


def cache_single_output(output, cache_file):
    if isinstance(output, csr_matrix):
        save_npz(cache_file, output)
    else:
        with open(cache_file, 'wb') as f:
            pickle.dump(output, f)
    print("Cached output to {}.".format(cache_file))

def load_single_cache_output(cache_file):
    if '.npz' in cache_file:
        output = load_npz(cache_file)
    else:
        with open(cache_file, 'rb') as f:
            output = pickle.load(f)

    print("Loaded cache from {}.".format(cache_file))
    return output

def manage_cache(cache_dir, extensions):

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if isinstance(extensions, str):
        single_output = True
        extensions = [extensions]
    else:
        single_output = False

    def decorator_manage_cache(func):
        def wrapper_cache_output(*args, **kwargs):
            run_id = "-".join([str(arg) for arg in args])

            cache_file = os.path.join(cache_dir, "{}_{}_{}{}".format(func.__name__, run_id, 0, extensions[0]))

            if not os.path.isfile(cache_file):
                outputs = func(*args, **kwargs)
                if single_output:
                    outputs = [outputs]

                for i, (output, extension) in enumerate(zip(outputs, extensions)):
                    cache_file = os.path.join(cache_dir, "{}_{}_{}{}".format(func.__name__, run_id, i, extension))
                    cache_single_output(output, cache_file)

            num_cache_files = len(glob.glob(cache_dir + "/{}_{}_*.".format(func.__name__, run_id)))

            outputs = []
            for i, extension in enumerate(extensions):
                cache_file = os.path.join(cache_dir, "{}_{}_{}{}".format(func.__name__, run_id, i, extension))
                output = load_single_cache_output(cache_file)
                outputs.append(output)

            if single_output:
                return outputs[0]
            return tuple(outputs)

        return wrapper_cache_output
    return decorator_manage_cache
