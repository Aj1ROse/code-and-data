import numpy as np


def VMD(signal, alpha, tau, K, DC, init, tol):


    # ---------- Preparations
    # Period and sampling frequency of input signal
    save_T = len(signal)
    fs = 1 / save_T

    # extend the signal by mirroring
    T = save_T
    f_mirror = np.zeros(2 * T)
    f_mirror[:T // 2] = signal[T // 2 - 1::-1]  # signal(T/2:-1:1)
    f_mirror[T // 2:3 * T // 2] = signal  # signal
    f_mirror[3 * T // 2:2 * T] = signal[T - 1:T // 2 - 1:-1]  # signal(T:-1:T/2+1)
    f = f_mirror

    # Time Domain 0 to T (of mirrored signal)
    T = len(f)
    t = np.arange(1, T + 1) / T  # (1:T)/T

    # Spectral Domain discretization
    freqs = t - 0.5 - 1 / T  # t-0.5-1/T

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = 500

    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)  # alpha*ones(1,K)

    # Construct and center f_hat
    f_hat = np.fft.fftshift(np.fft.fft(f))  # fftshift((fft(f)))
    f_hat_plus = f_hat.copy()  # f_hat
    f_hat_plus[:T // 2] = 0  # f_hat_plus(1:T/2) = 0

    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros((N, len(freqs), K), dtype=complex)  # zeros(N, length(freqs), K)

    # Initialization of omega_k
    omega_plus = np.zeros((N, K))  # zeros(N, K)
    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * i  # (0.5/K)*(i-1) -> (0.5/K)*i in Python (0-based)
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(K)))
    else:  # otherwise
        omega_plus[0, :] = 0

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros((N, len(freqs)), dtype=complex)  # zeros(N, length(freqs))

    # other inits
    uDiff = tol + np.finfo(float).eps  # tol+eps
    n = 0  # loop counter (Python从0开始，对应MATLAB的n=1)
    sum_uk = 0  # accumulator

    # ----------- Main loop for iterative updates
    while uDiff > tol and n < N - 1:  # not converged and below iterations limit

        # update first mode accumulator
        k = 0  # k = 1 in MATLAB, Python从0开始
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :,
                                                    0]  # u_hat_plus(n,:,K) + sum_uk - u_hat_plus(n,:,1)

        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / \
                                  (1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)

        # update first omega if not held at 0
        if not DC:
            # (freqs(T/2+1:T)*(abs(u_hat_plus(n+1, T/2+1:T, k)).^2)')/sum(abs(u_hat_plus(n+1,T/2+1:T,k)).^2)
            omega_plus[n + 1, k] = np.sum(freqs[T // 2:T] * (np.abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / \
                                   np.sum(np.abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

        # update of any other mode
        for k in range(1, K):  # k=2:K in MATLAB
            # accumulator
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            # mode spectrum
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / \
                                      (1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
            # center frequencies
            omega_plus[n + 1, k] = np.sum(freqs[T // 2:T] * (np.abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / \
                                   np.sum(np.abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

        # Dual ascent
        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * \
                               (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)  # sum(...,3) 对应 axis=1 (K维度)

        # loop counter
        n = n + 1

        # converged yet?
        uDiff = np.finfo(float).eps  # eps
        for i in range(K):  # i=1:K
            uDiff = uDiff + (1 / T) * np.sum((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]) * \
                                             np.conj(u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]))
        uDiff = abs(uDiff)

    # ------ Postprocessing and cleanup
    # discard empty space if converged early
    N = min(N, n + 1)  # min(N,n)
    omega = omega_plus[:N, :]  # omega_plus(1:N,:)

    # Signal reconstruction
    u_hat = np.zeros((T, K), dtype=complex)  # zeros(T, K)
    u_hat[T // 2:T, :] = u_hat_plus[N - 1, T // 2:T, :]  # u_hat((T/2+1):T,:) = squeeze(u_hat_plus(N,(T/2+1):T,:))
    u_hat[T // 2:0:-1, :] = np.conj(
        u_hat_plus[N - 1, T // 2:T, :])  # u_hat((T/2+1):-1:2,:) = squeeze(conj(u_hat_plus(N,(T/2+1):T,:)))
    u_hat[0, :] = np.conj(u_hat[-1, :])  # u_hat(1,:) = conj(u_hat(end,:))

    u = np.zeros((K, len(t)))  # zeros(K,length(t))

    for k in range(K):  # k = 1:K
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))  # real(ifft(ifftshift(u_hat(:,k))))

    # remove mirror part
    u = u[:, T // 4:3 * T // 4]  # u = u(:,T/4+1:3*T/4)

    # recompute spectrum
    u_hat_final = np.zeros((u.shape[1], K), dtype=complex)  # 重新计算最终的频谱
    for k in range(K):  # k = 1:K
        u_hat_final[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))  # fftshift(fft(u(k,:)))'

    return u, u_hat_final, omega 