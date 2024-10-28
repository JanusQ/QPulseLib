# import envelopehelpers as eh
import matplotlib.pyplot as plt
from common import envelopes as env
import numpy as np
from labrad.units import Unit

V, mV, us, ns, GHz, MHz, dBm, rad, uA = [
    Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad', 'uA')
]

edgeLens = [0.0 * ns, 0.0 * ns]
gateTime = 100 * ns
gatePadding = [9.0 * ns, 9.0 * ns]
w = 2.0
ripples = [0.0, 0.1, 0.0, 0.0]
tlist_xy = np.arange(60)


def wave_construction(gate_type):
    def xy_construction(start, piAmp, piLen, piFWHM, f10, fc, alpha, delta=-200 * MHz, piDf=0.0 * MHz, uwavePhase=0.0,
                        dAmp=0, dphase=0):
        def rotPulseHD(t0, piamp, w, phase=0, alpha=0.5, delta=-200 * MHz, state=1):
            """Rotation pulse using a gaussian envelope with half-derivative Y quadrature."""
            if state > 1: alpha = 0
            delta = 2 * np.pi * delta['GHz']
            x = env.gaussian(t0, w=w, amp=piamp, phase=phase)
            try:
                y = -float(alpha) * env.deriv(x) / delta
            except:
                print('========================')
                print(f'delta: {delta}')
                y = env.NOTHING
            return x + 1j * y

        df = f10 + piDf - fc
        # return rotPulseHD(start+piLen/2.0, piamp=piAmp+dAmp, w=piFWHM,phase=dphase, alpha=alpha, delta=delta)
        return env.mix(
            rotPulseHD(start + piLen / 2.0, piamp=piAmp + dAmp, w=piFWHM, phase=dphase, alpha=alpha, delta=delta),
            df) * np.exp(1j * uwavePhase)

    # def z_construction(start,piAmpz,fc,piDf=0.0*MHz,uwavePhase=0.0):
    #     def rotPulseZ(t0, angle=np.pi):
    #         """Rotation pulse using a gaussian envelope."""
    #         r = angle / np.pi
    #         w = piAmpz
    #         return env.gaussian(t0, w=w, amp=piAmpz * r)
    #
    #     df = f10 + piDf - fc
    #     # return rotPulseHD(start+piLen/2.0, piamp=piAmp+dAmp, w=piFWHM,phase=dphase, alpha=alpha, delta=delta)
    #     return env.mix(
    #         rotPulseZ(start + piLen / 2.0, angle = np.pi),
    #         df) * np.exp(1j * uwavePhase)

    # def cz_construction()

    def z_construction(start):
        def rotPulseHD(delta=-200 * MHz):
            delta = 2 * np.pi * delta['GHz']
            x = env.diabaticCZ(start, gateTime + edgeLens[0] + edgeLens[1],
                               0.2, w * ns,
                               ripples)
            try:
                y = -float(alpha) * env.deriv(x) / delta
            except:
                print('========================')
                print(f'delta: {delta}')
                y = env.NOTHING

            return x

        df = f10 - fc
        # return rotPulseHD(start+piLen/2.0, piamp=piAmp+dAmp, w=piFWHM,phase=dphase, alpha=alpha, delta=delta)
        # return env.mix(rotPulseHD())
        return rotPulseHD()

    piAmp = 0.5
    piLen = 60.0 * ns
    piFWHM = 30.0 * ns
    f10 = 5.0 * GHz
    fc = 5.1 * GHz
    alpha = 0.5
    delta = -200 * MHz
    dphase = 120
    tlist_xy = np.arange(60)
    tlist_cz = np.arange(120)
    tlist_i = np.zeros(70)

    if gate_type == 'I':
        wave_result = tlist_i
    elif gate_type in ['X', 'X/2', '-X/2', 'Y', 'Y/2', '-Y/2']:
        xy_wave = xy_construction(start=0, piAmp=piAmp, piLen=piLen, piFWHM=piFWHM, f10=f10, fc=fc, alpha=alpha,
                                  delta=delta, dphase=dphase)
        wave_result = xy_wave(np.arange(60))
    elif gate_type in ['Z', 'Z/2', '-Z/2']:
        z_wave = z_construction(start=0, piAmpz=0.5, fc=fc)
        wave_result = z_wave(tlist_xy)
    elif gate_type == 'cz':
        cz_wave = z_construction(start=-10)
        wave_result = cz_wave(tlist_cz)

    else:
        xy_wave = xy_construction(start=0, piAmp=piAmp, piLen=piLen, piFWHM=piFWHM, f10=f10, fc=fc, alpha=alpha,
                                  delta=delta, dphase=dphase)
        wave_result = xy_wave(tlist_xy)
    return wave_result


# To show the figure of wave generated from the above

# tlist = np.arange(-10, 110)
# wave_result = wave_construction('cz')
#
# wave_result = [wave_result.real, wave_result.imag]
# titleName = ['real', 'imag']
#
# for i in range(len(wave_result)):
#     fig = plt.figure()
#     plt.plot(tlist, wave_result[i], '.-', label='wave_result')
#
#     plt.legend()
# plt.show()
if __name__ == '__main__':
    tlist = np.arange(0, 60)
    wave_result = wave_construction('x')
    fig = plt.figure()
    for i in range(len(tlist)):
        plt.plot(tlist[:i + 1], wave_result.real[:i + 1], '.-', label='wave_result', color='blue')
        plt.xlim([-2, 62])
        plt.ylim([-0.5, 0.5])
        # plt.legend()
        # plt.close()
        plt.savefig(f'gif/{i}.jpg')
        # plt.show()
