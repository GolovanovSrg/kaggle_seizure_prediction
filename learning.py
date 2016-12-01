import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.io import loadmat
from os import listdir
from os.path import isfile, join, basename
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from essentia.standard import *
from yaafelib import *
from scipy.signal import butter, lfilter


def data2dict(dir_path):
    file_paths = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    info = pd.read_csv('info.csv', index_col='image')

    for path in file_paths:
        mat = loadmat(path)
        names = mat['dataStruct'].dtype.names
        dict = {n: mat['dataStruct'][n][0, 0] for n in names}
        file_name = basename(path)
        dict['class'] = info.ix[file_name, 'class']
        dict['file'] = file_name

        yield dict


def freq_content(dir_path):
    n_channels = 16
    signal_len = 240000
    fs = 400.0

    data = data2dict(dir_path)

    k = np.arange(signal_len)
    T = signal_len / fs
    frq = k / T
    frq = frq[list(range(signal_len // 2 + 1))]

    content0 = [np.zeros(signal_len // 2 + 1) for _ in range(n_channels)]
    content1 = [np.zeros(signal_len // 2 + 1) for _ in range(n_channels)]
    count0 = 0
    count1 = 0
    for d in tqdm(data):
        if d['class'] == 1:
            count1 += 1
        elif d['class'] == 0:
            count0 += 1

        for i in range(n_channels):
            y = d['data'][:, i]
            Y = np.abs(np.fft.rfft(y) / signal_len)
            Y = Y[list(range(signal_len // 2 + 1))]

            if d['class'] == 1:
                content1[i] += Y
            elif d['class'] == 0:
                content0[i] += Y

                # if count0 + count1 == 100:
                # break

    fig, ax = plt.subplots(n_channels, 2)

    for i in range(n_channels):
        ax[i, 0].plot(frq, content0[i] / count0)
        # ax[i, 0].set_xlabel('Freq (Hz)')
        # ax[i, 0].set_ylabel('|Y(freq)|')

    for i in range(n_channels):
        ax[i, 1].plot(frq, content1[i] / count1)
        # ax[i, 1].set_xlabel('Freq (Hz)')
        # ax[i, 1].set_ylabel('|Y(freq)|')

    plt.show()


def frame_gen(data, frame_size, step_size):
    offset = 0
    count = 0
    data_len = len(data)
    while offset + frame_size <= data_len:
        yield data[offset : offset + frame_size], count
        offset += step_size
        count += 1


def convert(dict):
    result = {}
    for feat_name in dict:
        array = dict[feat_name][0]
        if len(array) == 1:
            new_dict = {feat_name: array[0]}
        else:
            new_dict = {feat_name + '_' + str(i): array[i] for i in range(len(array))}
        result.update(new_dict)

    return result


def get_features(dir_path, out_path):
    frame_size = 12000
    step_size = frame_size
    sample_rate = 400

    n_coef_gfcc = 20

    n_channels = 16

    w = Windowing(type='hann')
    B, A = butter(5, 170.0 / (sample_rate / 2.0), btype='low')
    spectrum = Spectrum(size=frame_size)
    gfcc = GFCC(numberBands=100, numberCoefficients=n_coef_gfcc, inputSize=frame_size / 2 + 1,
                lowFrequencyBound=2, highFrequencyBound=200, sampleRate=sample_rate)
    max_mag_freq = MaxMagFreq(sampleRate=sample_rate)
    entropy = Entropy()
    envelope = Envelope(sampleRate=sample_rate)
    pitch_salience = PitchSalience(highBoundary=190, sampleRate=sample_rate)
    hfc = HFC(sampleRate=sample_rate, type='Jensen')
    flatnessSFX = FlatnessSFX()
    derSFX = DerivativeSFX()
    strong_peak = StrongPeak()
    flux = Flux()
    freq_bands = FrequencyBands(frequencyBands=np.array([1, 4, 8, 12, 30, 70, 180]), sampleRate=sample_rate)
    roll_off = RollOff(sampleRate=sample_rate)
    spec_cent = SpectralCentroidTime(sampleRate=sample_rate)
    spec_comp = SpectralComplexity(sampleRate=sample_rate)
    zcr = ZeroCrossingRate()
    min_to_total = MinToTotal()
    max_to_total = MaxToTotal()
    leq = Leq()
    compl = DynamicComplexity(sampleRate=sample_rate)
    lpc = LPC(order=15, sampleRate=sample_rate, type='warped')
    distr_shape = DistributionShape()
    cmoments = CentralMoments()
    loudness = Loudness()
    larm = Larm(sampleRate=sample_rate)
    rms = RMS()
    geo_mean = GeometricMean()
    ccor = CrossCorrelation(minLag=0, maxLag=0)

    fp = FeaturePlan(sample_rate=sample_rate)
    size_param = " blockSize={0} stepSize={0}".format(frame_size)
    fp.addFeature("obsi: OBSI" + size_param)
    fp.addFeature("lsf: LSF LSFNbCoeffs=15" + size_param)
    fp.addFeature("obsir: OBSIR" + size_param)
    fp.addFeature("percsharpness: PerceptualSharpness" + size_param)
    fp.addFeature("percspread: PerceptualSpread" + size_param)
    fp.addFeature("specslope: SpectralSlope" + size_param)
    fp.addFeature("specflat: SpectralFlatness" + size_param)
    df = fp.getDataFlow()
    engine = Engine()
    engine.load(df)

    write_header = True
    data = data2dict(dir_path)
    for d in tqdm(data):
        feats_data = pd.DataFrame()
        for i in range(n_channels):
            channel = d['data'][:, i].flatten()

            frame_list = list(frame_gen(channel, frame_size, step_size))
            for frame, cur_no in frame_gen(channel, frame_size, step_size):
                #frame = essentia.array(lfilter(B, A, frame, axis=0))
                spec = spectrum(w(frame))
                envel = envelope(frame)

                is_zero = np.sum(frame == 0.) > frame.shape[0] / 2.0

                feats_yaafe = {}
                if not is_zero:
                    row_data = np.require(frame.reshape(1, -1), requirements='C').astype(np.float64)
                    feats_yaafe = engine.processAudio(row_data)
                    feats_yaafe = convert(feats_yaafe)

                #ccor_coef = [ccor(frame, other[0])[0] for other in frame_list if cur_no != other[1]]
                #ccor_coef = np.mean(ccor_coef)

                log_freq = {'log_freq_' + str(i): 20 * np.log10(spec[i] + 1e-6) for i in range(1, 40)}

                lpc_coef, ref_coef = lpc(frame)
                lpc_coef = {'lpc_' + str(i): lpc_coef[i] for i in range(lpc_coef.shape[0])}
                ref_coef = {'ref_' + str(i): ref_coef[i] for i in range(ref_coef.shape[0])}

                moments = cmoments(frame)
                distr_coef = distr_shape(moments)
                distr_coef = {'distr_' + str(i): distr_coef[i] for i in range(len(distr_coef))}
                moments = {'moment_' + str(i): moments[i] for i in range(2, len(moments))}

                freq_energy = freq_bands(spec)
                freq_energy = freq_energy / (np.sum(freq_energy) + 1e-6)
                freq_energy = {'freq_energy_' + str(i): freq_energy[i] for i in range(freq_energy.shape[0])}

                _, coef_gfcc = gfcc(spec)
                coef_gfcc = coef_gfcc - np.mean(coef_gfcc)
                coef_gfcc = {'gfcc_' + str(i): coef_gfcc[i] for i in range(n_coef_gfcc)}

                feats = {'channel': i,
                         'file': d['file'],
                         'class': d['class'],
                         'is_zero': 1 if is_zero else 0,
                         'order': cur_no,
                         'max_mag_freq': max_mag_freq(spec),
                         'zcr': zcr(frame),
                         'min_to_total': min_to_total(envel),
                         'max_to_total': max_to_total(envel),
                         'entropy': entropy(spec),
                         'pitch_salience': pitch_salience(spec),
                         'hfc': hfc(spec),
                         'flatnessSFX': flatnessSFX(envel),
                         'maxDerBeforeMax': derSFX(envel)[1],
                         'derAvAfterMax': derSFX(envel)[0],
                         'leq': leq(frame),
                         'compl': compl(frame)[0],
                         'strong_peak': strong_peak(spec),
                         'skew': skew(spec),
                         'kurtosis': kurtosis(spec),
                         'var': np.var(spec),
                         'flux': flux(spec),
                         'roll_off': roll_off(spec),
                         'spec_cent': spec_cent(spec),
                         'spec_comp': spec_comp(spec),
                         'loudness': loudness(frame),
                         'larm' : larm(frame),
                         'rms': rms(frame),
                         'geo_mean': geo_mean(np.abs(frame))}#,
                         #'corr': ccor_coef}

                feats.update(coef_gfcc)
                feats.update(lpc_coef)
                feats.update(ref_coef)
                feats.update(moments)
                feats.update(distr_coef)
                feats.update(freq_energy)
                feats.update(feats_yaafe)
                feats.update(log_freq)

                feats_data = feats_data.append(feats, ignore_index=True)

        feats_data.to_csv(out_path,
                          header=write_header,
                          mode='a',
                          index=False,
                          dtype=np.float32)
        write_header = False


if __name__ == '__main__':
    dataset = 3

    train_dir_path = "/home/golovanov/contest/row_data/train_{}".format(dataset)
    feats_file_path = "/home/golovanov/contest/data/feats_{}.csv".format(dataset)

    # freq_content(train_dir_path)
    # count_obj(train_dir_path)
    get_features(train_dir_path, feats_file_path)

    test_dir_path = "home/golovanov/contest/row_data/test_{}".format(dataset)
    feats_file_path = "home/golovanov/contest/data/test_feats_{}.csv".format(dataset)
    #get_features(test_dir_path, feats_file_path)
