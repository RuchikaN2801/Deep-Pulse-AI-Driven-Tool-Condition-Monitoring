import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_spectrogram(signal, sr, save_path, file_name):
    plt.figure(figsize=(3, 3))

    # Convert to spectrogram
    S = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Plot
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.axis('off')

    # Save
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight', pad_inches=0)
    plt.close()


def process_dataset(signals, sr=22050, output_dir="data/spectrograms"):
    for i, signal in enumerate(signals):
        file_name = f"spec_{i}.png"
        generate_spectrogram(signal, sr, output_dir, file_name)
