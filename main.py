from music_class import MUSIC
import numpy as np
import matplotlib.pyplot as plt


def plot_signals(m):
    plt.figure(1)
    plt.subplot(211)
    plt.title("Modulated Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.plot(m.t, m.s[0])

    plt.subplot(212)
    plt.title("Steered Modulated Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    for n in range(m.N):
        plt.plot(m.t, m.X[n])

    plt.show()


def plot_music_range(angles_range):
    # frequencies = np.arange(2000, 10001, 1000)
    # frequencies = np.arange(10000, 20001, 1000)
    # frequencies = np.arange(20000, 30001, 1000)

    file = open("music_noisy_wideband_fm_1.txt", "w")
    # file.write("Different way of adding noise\n")
    file.write("AOA\t\t\t| Estimated AOA\n")
    file.write("---------------------------\n")
    # print("AOA\t| Estimated AOA")
    # print("-----------------------")

    # plt.figure(1)
    # plt.title(r"MUSIC Algorithm: $-90^\circ\leq\theta\leq90^\circ$")
    # plt.xlabel("Angle ($^\circ$)")
    # plt.ylabel("Power")
    # plt.subplot(211)

    # for freq in frequencies:
    # file.write("Secondary frequency = %.0f\n" % (freq))

    print("S", end="")
    for snr in range(1, 10, 1):
        file.write("SNR = %.0f\n" % (snr))
        file.write("+-------------------------+\n")

        for p in angles_range:
            print("*", end="")
            # m = MUSIC(p, wideband=True, f2=freq)
            m = MUSIC(p)
            m.calculateAOA(snr, fm_mod=True)
            # print(f"{p:.2f}\t| {m.aoa:.2f}")
            file.write("%.2f\t\t| %.2f\n" % (p, m.aoa))
            # plt.plot(m.angles * 180 / np.pi, 10 * np.log10(m.P_music))

        file.write("+-------------------------+\n")
        # print("+-------------------------+")

    # plt.plot(m.t, m.s[0])

    # plt.subplot(212)
    # plt.plot(m.t, m.s[0])
    # plt.show()


def test_accuracy(str_test_band, int_test_cntr):
    '''
    Test accuracy of algorithm for range of angles and averages RMSE (Root Mean Square Error)

    File format: "<AoA>,<average estimated AoA>,<average RMSE>\n"

    Range to test: -80 <= <theta> <= 80 in steps of 5

    Averages estimated AoA over <num_cycles> iterations:
    <est_aoa> = calculated by algorithm
    <avg_est_aoa> = 1./<num_cycles> * sum( <est_aoa> )

    Averages RMSE over <num_cycles> iterations:
    <rmse> = ( <est_aoa> - <aoa> )
    <avg_rmse> = sqrt( 1./<num_cycles> * sum( <rmse>**2 ) )

    Input:
     - <str_test_band>: String; bandwidth to test for; either "nb" or "wb"
     - <int_test_cntr>: Integer; external counter to differentiate files

    Output:
     - Text file with comma-separated values
    '''

    if (str_test_band == "wb"):
        bool_fm = True
    else:
        bool_fm = False

    filename = (
        "./output/accuracy/music_accuracy_%s_%d.csv") % (str_test_band, int_test_cntr)
    file_acc = open(filename, "w")
    test_angles = np.arange(-80, 81, 5)
    num_cycles = 100
    snr = 10
    total_est_aoa = 0
    total_rmse = 0

    for angle in test_angles:
        for n in range(num_cycles):
            m = MUSIC(angle)
            m.calculateAOA(snr, fm_mod=bool_fm)
            total_est_aoa += m.aoa
            total_rmse += (m.aoa - angle)**2

        avg_est_aoa = total_est_aoa / num_cycles
        avg_rmse = np.sqrt(total_rmse / num_cycles)

        file_acc.write("%.2f,%.2f,%.2f\n" % (angle, avg_est_aoa, avg_rmse))

    file_acc.close()


def test_resolution(fl_test_angle, int_test_snr, bool_test_fm):
    '''
    Test resolution for given angle(s) by plotting the output power spectrum

    Input:
     - <fl_test_angle>: Float; angle to test
     - <int_test_snr>: Integer; SNR to test for
     - <bool_test_fm>: Boolean; calculate AoA with or without FM modulation i.e. wideband or not

    Output:
     - Image file of graph
    '''

    filename = (
        "./output/resolution/music_resolution_%.2f_%d.png") % (fl_test_angle, int_test_snr)

    m = MUSIC(fl_test_angle)
    m.calculateAOA(int_test_snr, fm_mod=bool_test_fm)

    plt.figure(1)
    plt.title(r"MUSIC: AoA = $%.2f^\circ$" % (fl_test_angle))
    plt.xlabel("Angle/Direction of Arrival (degrees)")
    plt.ylabel("Power (dB)")
    plt.plot(m.angles * 180 / np.pi, m.P_music)
    plt.savefig(filename, bbox_inches="tight", format="png")


def test_efficiency(fl_test_angle, int_test_snr, bool_test_fm):
    '''
    Test efficiency i.t.o. time to calculate angle

    File format: "<iteration>,<execution time>\n"

    Input:
     - <fl_test_angle>: Float; angle to test
     - <int_test_snr>: Integer; SNR to test for
     - <bool_test_fm>: Boolean; calculate AoA with or without FM modulation i.e. wideband or not

    Output:
     - Image file of graph
    '''

    filename = (
        "./output/efficiency/music_efficiency_%.2f_%d.csv") % (fl_test_angle, int_test_snr)
    file_eff = open(filename, "w")

    for i in range(100):
        m = MUSIC(fl_test_angle)
        exec_time = m.calculateAOA(int_test_snr, fm_mod=bool_test_fm)
        file_eff.write("%d,%.3f\n" % (i + 1, exec_time))

    file_eff.close()


def test_sensitivity(fl_test_angle, bool_test_fm):
    '''
    Test accuracy for changes in SNR

    File format: "<snr_db>,<average RMSE>\n"

    SNR range to test (dB): -20 <= <snr_db> <= 20

    Averages RMSE over <num_cycles> iterations:
    <rmse> = ( <est_aoa> - <aoa> )
    <avg_rmse> = sqrt( 1./<num_cycles> * sum( <rmse>**2 ) )

    Input:
     - <fl_test_angle>: Float; angle to test
     - <bool_test_fm>: Boolean; calculate AoA with or without FM modulation i.e. wideband or not

    Output:
     - Text file with comma-separated values
    '''

    filename = (
        "./output/accuracy/music_accuracy_%s_%d.csv") % (str_test_band, int_test_cntr)
    file_acc = open(filename, "w")
    test_angles = np.arange(-80, 81, 5)
    num_cycles = 100
    snr = 10
    total_est_aoa = 0
    total_rmse = 0

    for angle in test_angles:
        for n in range(num_cycles):
            m = MUSIC(angle)
            m.calculateAOA(snr, fm_mod=bool_fm)
            total_est_aoa += m.aoa
            total_rmse += (m.aoa - angle)**2

        avg_est_aoa = total_est_aoa / num_cycles
        avg_rmse = np.sqrt(total_rmse / num_cycles)

        file_acc.write("%.2f,%.2f,%.2f\n" % (angle, avg_est_aoa, avg_rmse))

    file_acc.close()


# angle = 30
# music = MUSIC(angle)
# music.calculateAOA(100, fm_mod=True)
# music.plot_signals(music.X[0])
# plot_signals(music)
# print("AOA:\t\t%.2f" % (angle))
# print("Estimated AOA:\t%.3f" % (music.aoa))
# music.plot()

# print(np.exp(-1j * np.pi * np.sin(30 * np.pi / 180)))
# plot_music_range(np.arange(-90, 92, 2))
# Testing
# test_accuracy("nb", 0, False)
# test_resolution(30., 5, False)
test_efficiency(30., 5, False)
