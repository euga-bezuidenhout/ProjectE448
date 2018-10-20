class MUSIC:

    import numpy as np
    import matplotlib.pyplot as plt
    import time

    def __init__(self, theta_aoa):

        # angle of arrival (degrees)
        self.theta = self.np.pi / 180 * theta_aoa
        # self.wideband = wideband

        self.K = 1  # Number of signals
        self.N = 2   # Number of array elements
        self.M = 10000  # Number of signal snapshots
        self.F_s = 1. / 0.0001  # sampling frequency [Hz]
        self.Amp = 2  # Signal amplitude
        self.f1 = 10  # base signal frequency [Hz]
        # Speed of sound in Gordan's Bay @ ~15.5 C, 4% salinity:
        self.c = 1539.21294117647

        # additional signal frequency [Hz]
        # self.f2 = self.np.arange(2000, 10001, 1000)
        self.t = self.np.arange(self.M) / self.M  # time intervals

        # Steering vector
        self.__init_steering_vector()

        # Incoming signal(s)
        # self.__init_incoming_signal(wideband, f2)

        # Received signal
        # self.__init_received_signal(snr_dB)

    def __init_steering_vector(self):
        # K x N matrix => N x K later
        self.A = self.np.zeros((self.K, self.N), dtype=complex)
        self.d = self.c / (self.f1 * 2)  # distance between array elements

        for k in range(self.K):
            for n in range(self.N):
                self.A[k][n] = self.np.exp(-1j * 2 * self.np.pi *
                                           self.f1 * n * self.d *
                                           self.np.sin(self.theta) / self.c)
        self.A = self.A.T

    def calculateAOA(self, snr_dB, wideband=False, f2=0, fm_mod=False):
        if (fm_mod == False):
            self.__incoming_signal(wideband, f2)
        else:
            # print("Modulate signal!")
            self.__modulate_incoming_signal()

        self.__received_signal(snr_dB)
        start_time = self.__autocorrelation()
        self.__noise_subspace()
        end_time = self.__power_spectrum()

        return (end_time - start_time)

    def __incoming_signal(self, wb, f2):
        self.s = self.np.zeros((self.K, self.M), dtype=complex)

        if (wb == False):
            for k in range(self.K):
                sig = self.Amp * self.np.sin(2 * self.np.pi *
                                             self.f1 * self.t)
                self.s[k] = sig
        else:
            for k in range(self.K):
                sig = self.Amp * self.np.sin(2 * self.np.pi *
                                             self.f1 * self.t)
                sig += self.np.sin(2 * self.np.pi *
                                   f2 * self.t)
                self.s[k] = sig

    def __modulate_incoming_signal(self):
        self.s = self.np.zeros((self.K, self.M), dtype=complex)
        y_c = self.np.zeros((self.K, self.M), dtype=complex)

        # self.plt.figure(1)
        # self.plt.subplot(211)
        # self.plt.title("Modulator")
        # self.plt.ylabel("Amplitude")

        for k in range(self.K):
            sig = self.Amp * self.np.sin(2 * self.np.pi *
                                         self.f1 * self.t)
            self.s[k] = sig

        # self.plt.plot(self.t, self.s[0])

        # self.plt.subplot(212)
        # self.plt.ylabel("Amplitude")
        # self.plt.xlabel("Modulated Signal")

        # Carrier info
        f_c = 100
        m = 1.

        #  Modulated signal
        y_c = self.np.sin(2. * self.np.pi *
                          (self.t * f_c + m * self.s))

        # self.plt.plot(self.t, y_c[0])
        # self.plt.show()

        self.s = y_c

    def __received_signal(self, snr_dB):

        # print("SNR_dB: %.2f" % snr_dB)
        # print("Signal amplitude: %.2f" % self.Amp)
        P_sig_dB = 10 * self.np.log(self.Amp**2)
        # print("Signal power: %.2f" % P_sig_dB)
        P_noi_dB = P_sig_dB - snr_dB
        # print("Noise power: %.2f" % P_noi_dB)
        Amp_noi = self.np.sqrt(10**(P_noi_dB / 10))
        # print("Noise amplitude: %.2f" % Amp_noi)

        # white_noise = self.np.array(
        #     [self.np.random.normal(scale=0.5, size=self.M)], dtype=complex)
        white_noise = self.np.array(
            [self.np.random.normal(scale=Amp_noi, size=self.M)], dtype=complex)

        for w in range(1, self.N):
            # wn = self.np.array([self.np.random.normal(
            #     scale=0.5, size=self.M)], dtype=complex)
            wn = self.np.array([self.np.random.normal(
                scale=Amp_noi, size=self.M)], dtype=complex)
            white_noise = self.np.vstack((white_noise, wn))

        self.X = self.np.zeros((self.N, self.M), dtype=complex)
        # self.X = self.A * self.s
        self.X = self.A * self.s + white_noise
        # print(self.A.shape)
        # print(self.s.shape)
        # print(self.X.shape)

    def __autocorrelation(self):
        start_time = self.time.time()
        self.Rxx = self.np.cov(self.X, bias=True, rowvar=True)
        # print(self.Rxx)
        return start_time

    def __noise_subspace(self):
        U, S, V = self.np.linalg.svd(self.Rxx)
        self.U_n = U[:, (self.N - self.K)]
        # print(U.T)
        # print(S.T)
        # print(V.T)
        # print(self.U_n)

        # matlab_U = self.np.array([[-0.7071, 0.7071], [0.7071j, 0.7071j]])
        # self.U_n = self.np.array([0.7071, 0.7071j])

    def __power_spectrum(self):
        self.angles = self.np.linspace(-self.np.pi / 2, self.np.pi / 2, self.M)
        self.P_music = self.np.zeros(self.M)

        for m in range(self.M):
            p_shift = self.np.exp(-1j * 2 * self.np.pi *
                                  self.f1 * self.d *
                                  self.np.sin(self.angles[m]) / self.c)
            a_theta = self.np.array([1, p_shift])
            denom = self.np.abs(self.np.dot(a_theta.conj(), self.U_n))
            self.P_music[m] = 1. / (denom)

        self.angle_index = self.np.unravel_index(
            self.P_music.argmax(), self.P_music.shape)

        self.aoa = self.angles[self.angle_index] * 180 / self.np.pi
        # print(self.aoa)
        end_time = self.time.time()

        return end_time

    def plot_signals(self, signal):
        self.plt.title("Signal")
        self.plt.xlabel("Time")
        self.plt.ylabel("Amplitude")

        for k in range(self.K):
            self.plt.plot(self.t, signal)
        self.plt.show()

    def plot(self):
        self.plt.title(r"MUSIC Algorithm: $\theta$ = %.2f$^\circ$" %
                       (self.theta * 180 / self.np.pi))
        self.plt.xlabel("Angle ($^\circ$)")
        self.plt.ylabel("Power")
        self.plt.plot(self.angles * 180 / self.np.pi, 10 *
                      self.np.log10(self.P_music))
        self.plt.show()
