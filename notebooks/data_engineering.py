import pandas as pd
import numpy as np
from scipy import stats
import os

# librosa
import librosa
import librosa.display

import warnings



def columns() -> pd.MultiIndex:
    """
    Generate a multi-index column structure for feature statistics.

    Returns:
        pd.MultiIndex: A multi-index column structure with three levels: feature, statistics, and number.
    """
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12, tonnetz=6, mfcc=40, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1, spectral_contrast=7, spectral_rolloff=1,
                         mel_spectrogram = 128, delta_mfcc = 40, delta2_mfcc = 40, autocorrelation = 1,
                         tempogram = 384, onset_strength = 1)

    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    return columns.sort_values()

def compute_features(
    audio_path,
    year=None,
    rank=None,
    track_index=None
    ) -> pd.DataFrame:
    """
    Compute audio features for a given audio file.

    Parameters:
    - audio_path (str): The path to the audio file.
    - year (int, optional): The year of the audio file (default: None).
    - rank (int, optional): The rank of the audio file (default: None).
    - track_index (int, optional): The track index of the audio file (default: None).

    Returns:
    - features (pd.DataFrame): A DataFrame containing the computed audio features.

    Note:
    - This function uses the librosa library for audio processing.
    - The audio file should be in a format supported by librosa.
    """
    features = pd.Series(index=columns(), dtype=np.float32)
    warnings.filterwarnings('error', module='librosa')

    if year and rank:
        features["year"] = year
        features["rank"] = rank
    elif track_index:
        features["track_index"] = track_index

    def feature_stats(name, values):
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)

        # Use loc to ensure correct assignment in MultiIndex
        try:
            features.loc[(name, 'mean')] = np.mean(values, axis=1)
            features.loc[(name, 'std')] = np.std(values, axis=1)
            features.loc[(name, 'skew')] = stats.skew(values, axis=1)
            features.loc[(name, 'kurtosis')] = stats.kurtosis(values, axis=1)
            features.loc[(name, 'median')] = np.median(values, axis=1)
            features.loc[(name, 'min')] = np.min(values, axis=1)
            features.loc[(name, 'max')] = np.max(values, axis=1)
        except KeyError as e:
            print(f"KeyError encountered: {e}")
            print(f"Available feature columns: {features.index}")
        except Exception as e:
            print(f"Error processing {name}: {e}")

    try:
        filepath = audio_path
        x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

        feature_stats('onset_strength', np.abs(librosa.onset.onset_strength(y=x, sr=sr)).reshape(1, -1))
        feature_stats('zcr', librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512))
        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        feature_stats('chroma_cqt', librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7))
        feature_stats('chroma_cens', librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7))
        y = librosa.effects.harmonic(x)
        feature_stats('tonnetz', librosa.feature.tonnetz(y=y, sr=sr))

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1

        feature_stats('tempogram', librosa.feature.tempogram(y=x, sr=sr))
        feature_stats('chroma_stft', librosa.feature.chroma_stft(S=stft**2, n_chroma=12))
        feature_stats('rmse', librosa.feature.rms(S=stft))
        feature_stats('spectral_centroid', librosa.feature.spectral_centroid(S=stft))
        feature_stats('spectral_bandwidth', librosa.feature.spectral_bandwidth(S=stft))
        feature_stats('spectral_contrast', librosa.feature.spectral_contrast(S=stft, n_bands=6))
        feature_stats('spectral_rolloff', librosa.feature.spectral_rolloff(S=stft))

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2, n_mels=128, fmax=10000)
        feature_stats('mel_spectrogram', mel)
        feature_stats('mfcc', librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=40))
        feature_stats('delta_mfcc', librosa.feature.delta(librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=40)))
        feature_stats('delta2_mfcc', librosa.feature.delta(librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=40), order=2))
        feature_stats('autocorrelation', librosa.autocorrelate(x).reshape(1, -1))

    except Exception as e:
        pass

    return features


directory = 'Xiphorhynchus elegans/ElegantWoodcreeper/'

def validate_mp3_files(directory) -> list:
    """
    Validates the MP3 files in the given directory and returns a list of their locations.

    Args:
        directory (str): The directory to search for MP3 files.

    Returns:
        list: A list of file locations for the MP3 files found in the directory.
    """
    file_locations = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file[-4:] == ".mp3":
                file_locations.append(os.path.join(subdir, file))
    return file_locations

def process_features(
    dir_audio_files: list,
    table_name: str,
    dir_name: str = "."
) -> pd.DataFrame:
    """
    Process the features of audio files located at the given file locations and
    store the resulting DataFrame as a CSV file.

    Args:
        file_locations (list): A list of file locations of audio files.
        table_name (str): The name of the table to be used for the CSV file.
        dir_name (str, optional): The directory where the CSV file will be stored.
            Defaults to the current directory.

    Returns:
        pd.DataFrame: The processed features as a DataFrame.

    """
    file_locations = validate_mp3_files(dir_audio_files)
    series_list = []
    for i in file_locations:
        index = i[:-4].split("/")[-1]
        series_list.append(
            compute_features(
                track_index=index,
                audio_path=i
                )
            )

    df = pd.concat(series_list, axis=1).T
    df['track_index'] = pd.to_numeric(df['track_index'])
    df = df.set_index("track_index")
    df = df.sort_index()
    df.columns = ["_".join(x) for x in df.columns]
    df.to_csv(f"{dir_name}/{table_name}.csv")

    return df
