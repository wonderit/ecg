from __future__ import print_function
from builtins import range
from builtins import str
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
from tqdm import tqdm

import load
import decode
import util
from joblib import Memory


memory = Memory(cachedir='./data_cache', verbose=1)

def plot_confusion_matrix(cm, classes, model_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(util.get_confusion_figure_path(model_path))

@memory.cache
def get_model_predictions(args, x_val):
    from keras.models import load_model
    model = load_model(args.model_path)

    predictions = model.predict(x_val, verbose=1)[0]
    return predictions

def evaluate(args, params):
    dl = load.load(args, params)
    split = args.split
    x_val = dl.x_train if split == 'train' else dl.x_test
    y_val = dl.y_train if split == 'train' else dl.y_test
    mask_val = dl.mask_train if split == 'train' else dl.mask_val
    print("Size: " + str(len(x_val)) + " examples.")

    print("Predicting on:", split)
    predictions = get_model_predictions(args, [x_val, mask_val])

    if args.decode is True:
        language_model = decode.LM(dl.y_train, dl.output_dim, order=2)
        predictions = np.array([decode.beam_search(prediction, language_model)
                                for prediction in tqdm(predictions)])
    else:
        predictions = np.argmax(predictions, axis=-1)

    y_val_flat = np.argmax(y_val, axis=-1).flatten().tolist()
    predictions_flat = predictions.flatten().tolist()

    y_val_flat.extend(range(len(dl.classes)))
    predictions_flat.extend(range(len(dl.classes)))

    cnf_matrix = confusion_matrix(y_val_flat, predictions_flat).tolist()

    plot_confusion_matrix(
        np.log10(np.array(cnf_matrix) + 1),
        dl.classes,
        args.model_path)

    for i, row in enumerate(cnf_matrix):
        row.insert(0, dl.classes[i])

    print(classification_report(
        y_val_flat, predictions_flat,
        target_names=dl.classes))

    print(tabulate(cnf_matrix, headers=[c[:1] for c in dl.classes]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to files")
    parser.add_argument(
        "model_path",
        help="path to model, assuming prediction script generated")
    parser.add_argument("split", help="train/val", choices=['train', 'test'])
    parser.add_argument('--decode', action='store_true')
    args = parser.parse_args()
    params = util.get_model_params(args.model_path)
    evaluate(args, params)