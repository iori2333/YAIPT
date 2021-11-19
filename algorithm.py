import numpy as np
from numpy import ndarray


def plane(img: ndarray) -> ndarray:
    assert img.ndim == 2
    h, w = img.shape
    ret = np.ones((3 * h, 3 * w), dtype=np.uint8)
    for i in range(8):
        r, c = (7 - i) // 3, (7 - i) % 3
        ret[r * h: (r + 1) * h, c * w:(c + 1) * w] = \
            np.bitwise_and(img, 1 << i) >> i
    return ret


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
    k = 3

    def one_channel_denoise(channel: ndarray) -> ndarray:
        assert channel.ndim == 2

        padding = (k - 1) // 2
        padded = np.pad(channel, padding)

        @np.vectorize
        def get_median(x: int, y: int):
            return np.median(
                padded[x - padding:x + padding + 1, y - padding:y + padding + 1]
            )

        h, w = channel.shape
        xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        xx += padding
        yy += padding
        ret = get_median(xx, yy)
        return ret.astype(np.float)

    if img.ndim == 2:
        return one_channel_denoise(img)

    dst = img.copy()
    for i in range(img.ndim):
        dst[:, :, i] = one_channel_denoise(dst[:, :, i])
    return dst


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
    assert img.ndim == 2
    if img.dtype == np.float:
        img = (img * 255).astype(np.uint8)

    def img_filter(im: ndarray, kernel: np.ndarray):
        padding = (kernel.shape[-1] - 1) // 2
        padded = np.pad(im, padding)
        h, w = im.shape
        xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        xx += padding
        yy += padding

        @np.vectorize
        def core(x: int, y: int):
            return (padded[x - padding:x + padding + 1, y - padding:y + padding + 1] * kernel).sum()

        return core(xx, yy).astype(np.float)

    # Step.1 Gaussian filter
    gaussian_kernel = np.array([
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2],
    ]) / 159
    filtered = img_filter(img, gaussian_kernel).astype(np.uint8)

    # Step.2 Calc gradients and their directions
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=np.float)
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ], dtype=np.float)

    Gx = img_filter(filtered, sobel_x)
    Gy = img_filter(filtered, sobel_y)
    G = np.sqrt(Gx ** 2, Gy ** 2).astype(np.uint8)
    tangents = Gy / (Gx + 1e-12)
    theta = np.arctan(tangents)

    # Step.3 Non-max suppression
    gh, gw = G.shape
    dst = G[1:-1, 1:-1].copy()

    for i in range(1, gh - 1):
        for j in range(1, gw - 1):
            angle = theta[i, j]
            if angle > np.pi / 4:
                d_up = [0, 1]
                d_down = [1, 1]
                weight = 1 / tangents[i, j]
            elif angle >= 0:
                d_up = [1, 0]
                d_down = [1, 1]
                weight = tangents[i, j]
            elif angle >= - np.pi / 4:
                d_up = [1, 0]
                d_down = [1, -1]
                weight = -tangents[i, j]
            else:
                d_up = [0, -1]
                d_down = [1, -1]
                weight = -1 / tangents[i, j]

            gu1 = G[i + d_up[0], j + d_up[1]]
            gd1 = G[i + d_down[0], j + d_down[1]]
            gu2 = G[i - d_up[0], j - d_up[1]]
            gd2 = G[i - d_down[0], j - d_down[1]]

            gc1 = gu1 * weight + gd1 * (1 - weight)
            gc2 = gu2 * weight + gd2 * (1 - weight)

            if gc1 > G[i, j] or gc2 > G[i, j]:
                dst[i - 1, j - 1] = 0

    # Step.4 Two thresholds
    hist, bins = np.histogram(filtered.flatten(), 256)
    freq_sum = hist.cumsum()
    freq_sum = freq_sum / freq_sum[-1]
    th1, th2 = 0, 255
    for index, value in enumerate(freq_sum):
        if value > 0.4 and th1 == 0:
            th1 = index
        if value > 0.8:
            th2 = index
            break

    dst[dst < th1] = 0
    dst[dst >= th2] = 255

    dh, dw = dst.shape
    for i in range(1, dh - 1):
        for j in range(1, dw - 1):
            if 0 < dst[i, j] < 255:
                dst[i, j] = 255 \
                    if any([dst[i, j - 1] == 255,
                            dst[i, j + 1] == 255,
                            dst[i - 1, j] == 255,
                            dst[i + 1, j] == 255,
                            dst[i + 1, j + 1] == 255,
                            dst[i - 1, j - 1] == 255,
                            dst[i + 1, j - 1] == 255,
                            dst[i - 1, j + 1] == 255]) \
                    else 0
    return dst


def morphology(img: ndarray) -> ndarray:
    assert img.ndim == 2
    if img.dtype == np.float:
        img = (img * 255).astype(np.uint8)

    def img_filter(im: ndarray, padding: int, core_func):
        h, w = im.shape
        xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        xx += padding
        yy += padding

        return core_func(xx, yy).astype(np.float)

    def erode(im: ndarray) -> ndarray:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float)
        padding = (kernel.shape[-1] - 1) // 2
        padded = np.pad(im, padding)

        @np.vectorize
        def core(x: int, y: int):
            return (padded[x - padding:x + padding + 1, y - padding:y + padding + 1] * kernel).sum() >= 255

        mask = img_filter(im, padding, core).astype(np.bool)
        dst = im.copy()
        dst[mask] = 255
        return dst

    def dilate(im: ndarray) -> ndarray:
        kernel = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.float)
        padding = (kernel.shape[-1] - 1) // 2
        padded = np.pad(im, padding)

        @np.vectorize
        def core(x: int, y: int):
            return (padded[x - padding:x + padding + 1, y - padding:y + padding + 1] * kernel).sum() < 255 * 4

        mask = img_filter(im, padding, core).astype(np.bool)
        dst = im.copy()
        dst[mask] = 0
        return dst

    ret = erode(img)
    ret = dilate(ret)

    return ret
