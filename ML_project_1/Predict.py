import torch
import torchaudio
from cnn_ import AudioClassifier
from DataReading import UrbanSoundDataset
from Train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]

def predict(model, inp, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(inp)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == '__main__':
    cnn = AudioClassifier()
    state_dict = torch.load('guitar_model.pth')
    cnn.load_state_dict(state_dict)

    mel_spectorgram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectorgram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            'cpu')

    inp, target = usd[0][0], usd[0][1]
    inp.unsqueeze_(0)

