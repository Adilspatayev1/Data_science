import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from DataReading import UrbanSoundDataset
from Model import CNNNetwork
from cnn_ import AudioClassifier


BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = 'C:/Users/Adilbek/Downloads/Machine_Learning_project/Final_project_for_ML/Data/new_features_300_sec.csv'
ANNOTATIONS_FILE_TRAIN = 'C:/Users/Adilbek/Downloads/Machine_Learning_project/Final_project_for_ML/Data/train_data.csv'
ANNOTATIONS_FILE_TEST = 'C:/Users/Adilbek/Downloads/Machine_Learning_project/Final_project_for_ML/Data/test_data.csv'
AUDIO_DIR = 'C:/Users/Adilbek/Downloads/Machine_Learning_project/Final_project_for_ML/Data/wav_folds'
SAMPLE_RATE = 22050
NUM_SAMPLES = 661794

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    for inp, targets in data_loader:
        inputs, targets = inp.to(device), targets.to(device)

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

        _, prediction = torch.max(predictions, 1)
        correct_prediction += (prediction == targets).sum().item()
        total_prediction += prediction.shape[0]
    num_batches = len(data_loader)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Loss: {avg_loss}, Accuracy: {acc:.2f}')

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i + 1}')
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print('-----------------------------')
    print('Training is done')

def train_pipline(cnn, device):
    mel_spectorgram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    usd = UrbanSoundDataset(ANNOTATIONS_FILE_TRAIN,
                            AUDIO_DIR,
                            mel_spectorgram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    train_data_loader = create_data_loader(usd, BATCH_SIZE)
    loss_fn = nn.KLDivLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)
    torch.save(cnn.state_dict(), 'guitar_model.pth')
    print('Model trained and stored at guitar_model.pth')

def test_pipline(cnn, device, state_dict):
    cnn.load_state_dict(state_dict)
    cnn.eval()
    mel_spectorgram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    usd = UrbanSoundDataset(ANNOTATIONS_FILE_TEST,
                            AUDIO_DIR,
                            mel_spectorgram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    test_data_loader = create_data_loader(usd, BATCH_SIZE)

    correct_prediction = 0
    total_prediction = 0

    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_mean, inputs_std = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_mean)/inputs_std

            outputs = cnn(inputs)

            _, prediction = torch.max(outputs, 1)

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using {device} device')

    cnn = AudioClassifier().to(device)
    print(cnn)

    train_pipline(cnn, device)
    state_dict = torch.load('guitar_model.pth')
    cnn = AudioClassifier().to(device)
    test_pipline(cnn, device, state_dict)

