import numpy as np
from numpy import ndarray


def plane(img: ndarray) -> ndarray:
    return None


def equalize(img: ndarray) -> ndarray:
    def one_channel_equalize(channel: ndarray) -> ndarray:
        assert channel.ndim == 2
        hist, bins = np.histogram(channel.flatten(), 256)
        freq_sum = hist.cumsum()
        freq_sum = freq_sum / freq_sum[-1]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        out = np.interp(channel.flat, bin_centers, freq_sum)
        ret = out.reshape(channel.shape)
        return ret

    if img.ndim == 2:
        return one_channel_equalize(img)

    dst = img.copy()
    for i in range(img.ndim):
        dst[:, :, i] = one_channel_equalize(dst[:, :, i])
    return dst


def denoise(img: ndarray) -> ndarray:
    return None


def interpolate(img: ndarray) -> ndarray:
    if img.ndim == 2:
        img = img.reshape(*img.shape, 1)

    h, w, c = img.shape
    n = 2
    dst_h, dst_w = h * n, w * n

    xx, yy = np.meshgrid(np.arange(dst_h), np.arange(dst_w), indexing='ij')
    ret = img[xx // n, yy // n, :]
    return ret


def dft(img: ndarray) -> ndarray:
    assert img.ndim == 2
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return np.log(np.abs(img_fft))


def butterworth(img: ndarray) -> ndarray:
    assert img.ndim == 2
    h, w = img.shape

    n, d0 = 2, 20
    vc, hc = h // 2, w // 2

    xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    d = np.sqrt((xx - vc) ** 2 + (yy - hc) ** 2)
    mask = 1 / (1 + (d / d0) ** (2 * n))

    dft_img = np.fft.fft2(img)
    dft_img = np.fft.fftshift(dft_img)
    filtered = dft_img * mask
    ret = np.fft.ifftshift(filtered)
    ret = np.fft.ifft2(ret)
    return np.abs(ret)


def canny(img: ndarray) -> ndarray:
    return None


def morphology(img: ndarray) -> ndarray:
    return None
