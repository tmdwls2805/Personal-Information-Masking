import numpy as np

def modelsave(model, model_save, model_save_weights):
    model_json = model.to_json()
    with open(model_save, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_save_weights)
    print("Saved model to disk")


def numpy_save(saves, names, data_save):
    for idx, item in enumerate(saves):
        np.save(data_save .format(names[idx]), item)
    return