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


def test_accuracy(int_test_snr, bool_test_fm):
    '''
    Test accuracy of algorithm for range of angles and averages RMSE (Root Mean Square Error)

    File format: "<AoA> & <average estimated AoA> & <average RMSE>\\\n"

    Range to test: -80 <= <theta> <= 80 in steps of 5

    Averages estimated AoA over <num_cycles> iterations:
    <est_aoa> = calculated by algorithm
    <avg_est_aoa> = 1./<num_cycles> * sum( <est_aoa> )

    Averages RMSE over <num_cycles> iterations:
    <rmse> = ( <est_aoa> - <aoa> )
    <avg_rmse> = sqrt( 1./<num_cycles> * sum( <rmse>**2 ) )

    Input:
     - <int_test_snr>: Integer; SNR to test for
     - <bool_test_fm>: Boolean; calculate AoA with or without FM modulation i.e. wideband or not

    Output:
     - Text file with LaTeX table-ready values
     - Image file with average RMSE vs AoA
    '''

    if (bool_test_fm):
        band = "wb"
    else:
        band = "nb"

    filename = "./output/accuracy/music_accuracy_%s_%d.txt" % (
        band, int_test_snr)
    filename_img = "./output/accuracy/music_accuracy_%s_%d.png" % (
        band, int_test_snr)
    file_acc = open(filename, "w")
    test_angles = np.arange(-90, 91, 5)
    num_cycles = 100
    arr_avg_rmse = np.zeros(test_angles.size)

    for i in range(test_angles.size):
        total_rmse = 0
        total_est_aoa = 0

        for n in range(num_cycles):
            m = MUSIC(test_angles[i])
            m.calculateAOA(int_test_snr, fm_mod=bool_test_fm)
            total_est_aoa += m.aoa
            total_rmse += (m.aoa - test_angles[i])**2

        avg_est_aoa = total_est_aoa / num_cycles
        avg_rmse = np.sqrt(total_rmse / num_cycles)
        arr_avg_rmse[i] = avg_rmse

        file_acc.write("%.2f & %.2f & %.2f" %
                       (test_angles[i], avg_est_aoa, avg_rmse))

        if ((i + 1) % 2 == 0):
            file_acc.write("\\\\\n")
        else:
            file_acc.write(" & ")

    file_acc.close()

    plt.figure(1)
    plt.title("MUSIC Algorithm: Accuracy of Algorithm")
    plt.xlabel("Angle of Arrival (degrees)")
    plt.ylabel("Average RMSE (dB)")
    plt.plot(test_angles, 10 * np.log(arr_avg_rmse))
    plt.savefig(filename_img, bbox_inches="tight", format="png")


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

    if (bool_test_fm):
        band = "wb"
    else:
        band = "nb"

    filename = (
        "./output/resolution/music_resolution_%s_%.2f_%d.png") % (band, fl_test_angle, int_test_snr)
    num_cycles = 100
    test_angles = np.linspace(-90, 90, 10000)
    avg_P = np.zeros(test_angles.size)

    for n in range(num_cycles):
        m = MUSIC(fl_test_angle)
        time = m.calculateAOA(int_test_snr, fm_mod=bool_test_fm)
        avg_P += m.P_music

    avg_P /= num_cycles

    plt.figure(1)
    plt.title(r"MUSIC: AoA = $%.2f^\circ$" % (fl_test_angle))
    plt.xlabel("Angle/Direction of Arrival (degrees)")
    plt.ylabel("Power (dB)")
    plt.plot(test_angles, avg_P)
    plt.savefig(filename, bbox_inches="tight", format="png")


def test_efficiency(fl_test_angle, int_test_snr, bool_test_fm):
    '''
    Test efficiency i.t.o. time to calculate angle

    File format: "<iteration> & <execution time> & <iteration> & <execution time> & <iteration> & <execution time>\\\n"

    Input:
     - <fl_test_angle>: Float; angle to test
     - <int_test_snr>: Integer; SNR to test for
     - <bool_test_fm>: Boolean; calculate AoA with or without FM modulation i.e. wideband or not

    Output:
     - Image file of graph
    '''

    if (bool_test_fm):
        band = "wb"
    else:
        band = "nb"

    filename = (
        "./output/efficiency/music_efficiency_%s_%.2f_%d.txt") % (band, fl_test_angle, int_test_snr)
    file_eff = open(filename, "w")
    exec_times = np.zeros(102)
    exec_times_100 = np.zeros(100)

    for i in range(exec_times.size):
        m = MUSIC(fl_test_angle)
        exec_times[i] = m.calculateAOA(int_test_snr, fm_mod=bool_test_fm)

        if ((i + 1) % 3 == 0):
            file_eff.write("%d & %.3f & %d & %.3f & %d & %.3f\\\\\n" % (
                i - 1, exec_times[i - 2], i, exec_times[i - 1], i + 1, exec_times[i]))

    exec_times_100 = exec_times[0:100]
    file_eff.write("min,%.3f\n" % (np.amin(exec_times_100)))
    file_eff.write("max,%.3f\n" % (np.amax(exec_times_100)))
    file_eff.write("avg,%.3f\n" % (np.average(exec_times_100)))
    file_eff.close()


def test_sensitivity(fl_test_angle, bool_test_fm):
    '''
    Test accuracy for changes in SNR

    File format: "<snr_db>,<average RMSE>\n"

    SNR range to test (dB): -50 <= <snr_db> <= 40 in steps of 5

    Averages RMSE over <num_cycles> iterations:
    <rmse> = ( <est_aoa> - <aoa> )
    <avg_rmse> = sqrt( 1./<num_cycles> * sum( <rmse>**2 ) )

    Input:
     - <fl_test_angle>: Float; angle to test
     - <bool_test_fm>: Boolean; calculate AoA with or without FM modulation i.e. wideband or not

    Output:
     - Text file with LaTeX table-ready values
     - Image file with graph of RMSE-vs-SNR
    '''

    if (bool_test_fm):
        band = "wb"
    else:
        band = "nb"

    filename = (
        "./output/sensitivity/music_sensitivity_%s_%.2f.txt") % (band, fl_test_angle)
    filename_img = (
        "./output/sensitivity/music_sensitivity_%s_%.2f.png") % (band, fl_test_angle)
    file_sens = open(filename, "w")
    test_snr = np.arange(-50, 41, 5)
    num_cycles = 100
    arr_avg_rmse_db = np.zeros(test_snr.size)

    for i in range(test_snr.size):
        total_rmse = 0

        for n in range(num_cycles):
            m = MUSIC(fl_test_angle)
            exec_time = m.calculateAOA(test_snr[i], fm_mod=bool_test_fm)
            total_rmse += (m.aoa - fl_test_angle)**2

        avg_rmse_db = 10 * np.log(np.sqrt(total_rmse / num_cycles))
        arr_avg_rmse_db[i] = avg_rmse_db

        file_sens.write("%d & %.4f" % (test_snr[i], avg_rmse_db))

        if (((i + 1) % 3) == 0):
            file_sens.write("\\\\\n")
        else:
            file_sens.write(" & ")

    file_sens.close()

    plt.figure(1)
    plt.title("MUSIC Algorithm: Sensitivity to Noise")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average RMSE (dB)")
    plt.plot(test_snr, arr_avg_rmse_db)
    plt.savefig(filename_img, bbox_inches="tight", format="png")


# angle = 30
# music = MUSIC(angle)
# t = music.calculateAOA(20, fm_mod=False)
# music.plot_signals(music.X[0])
# plot_signals(music)
# print("AOA:\t\t%.2f" % (angle))
# print("Estimated AOA:\t%.3f" % (music.aoa))
# music.plot()

# print(np.exp(-1j * np.pi * np.sin(30 * np.pi / 180)))
# plot_music_range(np.arange(-90, 92, 2))

# Testing Narrowband
# test_accuracy(0, False)  # Done
# test_resolution(30., 5, False) # Done
# test_efficiency(30., 5, False) # Done
# test_sensitivity(30., False)  # Done

# Testing Wideband
# test_accuracy(0, True) # Done
# test_resolution(30., 5, True) # Done
# test_efficiency(30., 5, True) # Done
test_sensitivity(30., True)
