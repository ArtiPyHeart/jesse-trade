import numpy as np
import scipy.signal as signal


def nrbo(imf: np.ndarray, max_iter: int = 10, tol: float = 1e-6) -> np.ndarray:
    for _ in range(max_iter):
        ### 计算极值点
        peaks, _ = signal.find_peaks(imf)
        valleys, _ = signal.find_peaks(-imf)
        extrema = np.sort(np.concatenate((peaks, valleys)))

        if len(extrema) < 2:
            break

        ### 计算边界点的一阶和二阶导数
        left_extrema = extrema[0]
        right_extrema = extrema[-1]

        ### 一阶导数
        df_left = (imf[left_extrema + 1] - imf[left_extrema - 1]) / 2
        df_right = (imf[right_extrema + 1] - imf[right_extrema - 1]) / 2

        ###  二阶导数
        d2f_left = imf[left_extrema + 1] - 2 * imf[left_extrema] + imf[left_extrema - 1]
        d2f_right = (
            imf[right_extrema + 1] - 2 * imf[right_extrema] + imf[right_extrema - 1]
        )

        ###  NRBO迭代更新边界点
        new_left = imf[left_extrema] - df_left / d2f_left
        new_right = imf[right_extrema] - df_right / d2f_right

        ###  判断收敛
        if (
            np.abs(new_left - imf[left_extrema]) < tol
            and np.abs(new_right - imf[right_extrema]) < tol
        ):
            break

        imf[left_extrema] = new_left
        imf[right_extrema] = new_right

    return imf
