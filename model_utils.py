import matplotlib.pyplot as plt
from datetime import datetime

def save_plot_accuracy_loss(history, model_name, filename):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.savefig(f'fixtures/models/{filename}')


# Save the model for later use
def save_model(model, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f'models_test/{model_name}_{timestamp}.keras')