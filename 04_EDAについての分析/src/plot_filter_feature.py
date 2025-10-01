import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift


def create_filter_response(filter_type, fs=4.0, **kwargs):
    """フィルタの周波数応答を作成"""

    if filter_type == "butterworth_lowpass":
        cutoff = kwargs['cutoff']
        order = kwargs.get('order', 4)
        nyquist = fs / 2
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        w, h = signal.freqz(b, a)
        freq_hz = (w / np.pi) * (fs / 2)
        return freq_hz, h, cutoff

    elif filter_type == "butterworth_highpass":
        cutoff = kwargs['cutoff']
        order = kwargs.get('order', 4)
        nyquist = fs / 2
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        w, h = signal.freqz(b, a)
        freq_hz = (w / np.pi) * (fs / 2)
        return freq_hz, h, cutoff

    elif filter_type == "butterworth_bandpass":
        low_cutoff = kwargs['low_cutoff']
        high_cutoff = kwargs['high_cutoff']
        order = kwargs.get('order', 4)
        nyquist = fs / 2
        low_norm = min(low_cutoff / nyquist, 0.99)
        high_norm = min(high_cutoff / nyquist, 0.99)
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        w, h = signal.freqz(b, a)
        freq_hz = (w / np.pi) * (fs / 2)
        return freq_hz, h, (low_cutoff, high_cutoff)

    elif filter_type == "gaussian":
        sigma = kwargs.get('sigma', 2.0)   # [samples]
        size  = kwargs.get('size', 63)     # odd
        n = np.arange(size) - (size - 1)/2
        b = np.exp(-0.5 * (n / sigma) ** 2)
        b /= b.sum()                       # 直流ゲイン=1に正規化
        w, h = signal.freqz(b, 1, worN=2048, fs=fs)
        freq_hz = w
        mag_db = 20*np.log10(np.maximum(np.abs(h), 1e-12))
        cutoff_idx = np.argmin(np.abs(mag_db + 3))
        actual_cutoff = freq_hz[cutoff_idx]
        return freq_hz, h, actual_cutoff

    elif filter_type == "moving_average":
        window_size = kwargs.get('window_size', 5)

        # 移動平均フィルタの係数
        b = np.ones(window_size) / window_size
        a = 1
        w, h = signal.freqz(b, a)
        freq_hz = (w / np.pi) * (fs / 2)

        # 実際の-3dB点を計算してカットオフとして返す
        magnitude_db = 20 * np.log10(np.abs(h))
        cutoff_idx = np.argmin(np.abs(magnitude_db + 3))
        actual_cutoff = freq_hz[cutoff_idx]

        return freq_hz, h, actual_cutoff

    elif filter_type == "ideal_lowpass":
        cutoff = kwargs['cutoff']

        # 理想ローパスフィルタ（矩形特性）
        freq_hz = np.linspace(0, fs/2, 1000)
        h = np.where(freq_hz <= cutoff, 1.0, 0.0)
        h = h.astype(complex)  # 複素数として返す
        return freq_hz, h, cutoff

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def plot_filter_characteristics(filter_type, fs=4.0, output_dir="/home/mori/projects/affective-forecast/plots/フィルタ特性図", **kwargs):
    """フィルタ特性図をプロット・保存"""

    # フィルタ応答を作成
    freq_hz, h, cutoff_info = create_filter_response(filter_type, fs=fs, **kwargs)

    # ファイル名作成
    if isinstance(cutoff_info, tuple):
        filename = f"{filter_type}_low{cutoff_info[0]}_high{cutoff_info[1]}_fs{fs}"
    else:
        filename = f"{filter_type}_cutoff{cutoff_info}_fs{fs}"

    # 出力ディレクトリ作成
    filter_dir = os.path.join(output_dir, filter_type)
    os.makedirs(filter_dir, exist_ok=True)

    # プロット
    plt.figure(figsize=(10, 6))

    # 振幅特性
    plt.subplot(2, 1, 1)
    plt.plot(freq_hz, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
    plt.title(f'{filter_type.replace("_", " ").title()} Filter - Magnitude Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.xlim(0, fs/2)

    # -3dBラインを追加
    plt.axhline(y=-3, color='red', linestyle='--', alpha=0.7, label='-3dB')

    # カットオフ周波数の破線を追加
    if isinstance(cutoff_info, tuple):
        plt.axvline(x=cutoff_info[0], color='orange', linestyle='--', alpha=0.8,
                   label=f'Low cutoff: {cutoff_info[0]}Hz')
        plt.axvline(x=cutoff_info[1], color='orange', linestyle='--', alpha=0.8,
                   label=f'High cutoff: {cutoff_info[1]}Hz')
    else:
        plt.axvline(x=cutoff_info, color='orange', linestyle='--', alpha=0.8,
                   label=f'Cutoff: {cutoff_info}Hz')
    plt.legend()

    # 位相特性
    plt.subplot(2, 1, 2)
    plt.plot(freq_hz, np.angle(h))
    plt.title(f'{filter_type.replace("_", " ").title()} Filter - Phase Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [radians]')
    plt.grid(True)
    plt.xlim(0, fs/2)

    # カットオフ周波数の破線を位相特性にも追加
    if isinstance(cutoff_info, tuple):
        plt.axvline(x=cutoff_info[0], color='orange', linestyle='--', alpha=0.8)
        plt.axvline(x=cutoff_info[1], color='orange', linestyle='--', alpha=0.8)
    else:
        plt.axvline(x=cutoff_info, color='orange', linestyle='--', alpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(filter_dir, f"{filename}.svg"), bbox_inches='tight')
    plt.close()

    return filename


if __name__ == "__main__":
    output_path = "/home/mori/projects/affective-forecast/plots/フィルタ特性図"
    fs = 4.0

    plot_filter_characteristics("butterworth_lowpass", fs=fs, cutoff=0.25, order=4)
    plot_filter_characteristics("butterworth_highpass", fs=fs, cutoff=0.25, order=4)
    plot_filter_characteristics("butterworth_bandpass", fs=fs, low_cutoff=0.2, high_cutoff=1.0, order=4)
    plot_filter_characteristics("gaussian", fs=fs, sigma=2.0, size=64)
    plot_filter_characteristics("moving_average", fs=fs, window_size=5)
    plot_filter_characteristics("ideal_lowpass", fs=fs, cutoff=0.25)