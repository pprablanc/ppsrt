# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:30:11 2017

@author: PRABLANC P.
"""

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt


class ProsodicModificationRealTime(object):
    """Modification of the pitch of the voice with or without preservation of the spectrale envelope




    Notes: resampling step is not handled properly leading to "clic" sounds. The scipy.resample()
        function works with a Fourier-domain method which makes it impossible to deal with boundaries
        issues. It would be necessary to incorporate in the frame to resample a few samples before and after
        in order to deal with boundarie issues. A time-domain-based method would be probably more appropriate
        such as scipy.resample_poly()
    """

    def __init__(self, fs=16000.0, frame_format='Float32'):
        self.fs = np.float(fs)
        self._pitch_rate = 1.0
        self.MIN_INTENSITY_VALUE = -40
        self.fs_norm = self.fs / 16000.0
        self._mid_buffer_size = int(256.0 * self.fs_norm)
        self._buffer = 0.001*np.random.rand(5 * self._mid_buffer_size, 1)
        self.win_length = int(128.0 * self.fs_norm)  # based on fs = 16 kHz
        win = np.hanning(self.win_length*2)
        self.win_G = np.reshape(win[0:self.win_length], [self.win_length, 1])
        self.win_D = np.reshape(win[self.win_length:], [self.win_length, 1])
        self.win_s = int(256*self.fs_norm)

        # initialization related to pitch change
        self.initialize(self._pitch_rate, frame_format)

    def initialize(self, pitch_rate, frame_format='Float32'):
        #==============================================================================
        # Pitch-shifting initialization
        #==============================================================================
        self.frame_format = frame_format
        # if pitch_rate < 1, the weights can't overflow the pointer
        self.length_weight = int(120 * self.fs_norm)  # length of the weight window
        diff = np.ceil(self.win_s * pitch_rate) - self.length_weight
        if diff < 0:
            self.length_weight = int(self.length_weight + diff)
        tmp = np.arange(-self.length_weight, self.length_weight+1)
        # weight used in frame intercorrelation
        self.weight = np.ones(self.length_weight * 2 + 1)
        self.weight[0:self.length_weight] = 1 - np.abs(tmp[0:self.length_weight]) / float(self.length_weight) * 0.5
        self.weight[self.length_weight+1:] = 1 - np.abs(tmp[self.length_weight+1:]) / float(self.length_weight) * 0.5
        self.weight = self.weight.reshape([self.length_weight * 2 + 1, 1])
        self.frame_length = 1 + int(np.ceil(self.win_s * pitch_rate)) + 2*self.length_weight + self.win_s - 1
        self.shift_mod = int(np.ceil(self.win_s*pitch_rate))
        self.zeros_frame_mod = np.zeros([self.shift_mod, 1])
        if pitch_rate < 1:
            self.buffer_mod = np.zeros([5 * self.win_s, 1])
            print(pitch_rate)
        else:
            self.buffer_mod = np.zeros([5 * self.shift_mod, 1])

        #==============================================================================
        # LPC initialization
        #==============================================================================
        self.alpha = 0.97
        self.n_lpc = 16
        self.w_hamming = np.hamming(self.shift_mod)


    def pitchshifting(self, new_frame):
        new_frame = new_frame.reshape([new_frame.size, 1])
        self._buffer = self.buffershift(self._buffer,new_frame)

        frame_mod_prev = \
            self.buffer_mod[(self.shift_mod + 1 - self.length_weight) : (1 + self.shift_mod + self.length_weight + self.win_s)]

        frame_s = np.reshape(self._buffer[0:self.frame_length], [self.frame_length, 1])
        intensity = 10 * np.log10(np.sum(frame_s**2) + 1e-30)

        if intensity > self.MIN_INTENSITY_VALUE:
            ic = sp.correlate(frame_mod_prev, frame_s[0:self.win_s])
            intercorr = ic[self.win_s-2:self.win_s-1+2*self.length_weight] * self.weight
            I = intercorr.argmax()
            k_pos = I-(self.length_weight+1)
        else:
            k_pos = 0
        # Not clean, indices overflow buffer_mod size.
        # It works because overflowed values are taken into account.
        self.buffer_mod[self.shift_mod+k_pos+self.win_length:self.shift_mod+k_pos+self.frame_length] = \
            frame_s[self.win_length:self.frame_length]
        self.buffer_mod[self.shift_mod+k_pos:self.shift_mod+k_pos+self.win_length] = \
            + self.buffer_mod[self.shift_mod+k_pos:self.shift_mod+k_pos+self.win_length]*self.win_D \
            + frame_s[0:self.win_length]*self.win_G

        # shift modified buffer
        self.buffer_mod = self.buffershift(self.buffer_mod,self.zeros_frame_mod)
#        frame_residue = lpcanalysis(self.buffer_mod[:self.shift_mod])
        # resample
        frame_output = sp.resample(self.buffer_mod[:self.shift_mod], new_frame.size)
        return frame_output

    def lpcanalysis(self, frame):
        raise NotImplementedError('Not yet implemented');
#        frame_preemphasis = sp.lfilter([1, -self.alpha], 1, frame.reshape([frame.size]))
#        frame_preemphasis_windowed = frame_preemphasis[self.n_lpc:] \
#            *self.w_hamming
#        R = sp.correlate(frame_preemphasis_windowed, frame_preemphasis_windowed)  # coefficients de corrélation
#        Ri = R[window_length-1:window_length+self.n_lpc]
#
#        lpc_durbin = self.durbin(Ri)  # calcul des ai
#        ai = lpc_durbin['a']
#        ai = ai.reshape([ai.size])
#        self._lpc_coeff[:, n] = ai
#        frame_filt = sp.lfilter(ai, 1.0, frame_preemphasis)  # filtrage du signal
#        frame_residue = frame_filt[self.n_lpc:]


    def buffershift(self, buffer_loc, frame):
        buffer_loc[:-frame.size] = buffer_loc[frame.size:]
        buffer_loc[-frame.size:] = frame.reshape([frame.size, 1])
        return buffer_loc

    def str2numpy(self, frame):
        return np.fromstring(frame,self.frame_format)

    def numpy2str(self, frame):
        frame = frame.astype(self.frame_format)
        return frame.tostring()


    def durbin(self,frame_corr):
        """Perform linear predictive coding with Levinson-Durbin recursion.

        Args:
            frame_corr (numpy array): First samples of auto-correlation frame.

        Returns:
            {
                'a': a, the filter coefficient of an auto-regressive model
                'k': k, the reflexion coefficient
                'En': En, the prediction error
            }
        """
        R0 = frame_corr[0]
        frame_corr = np.reshape(frame_corr/R0, [frame_corr.size, 1])
        p = frame_corr.size - 1
        k = np.zeros([p, 1])
        a = 1
        for n in range(0, p):
            a = np.append(a, 0.)
            a = a.reshape([a.size, 1])
            r = frame_corr[0:n+2]
            En = np.sum(r*a)
            Bn = np.sum(np.flipud(r)*a)
            ki = -Bn/En
            a = a + np.flipud(a)*ki
            k[n] = ki
        En = R0*np.sum(frame_corr*a)
        return {
            'a': a,
            'k': k,
            'En': En,
            }


    def lsf2poly(lsf):
        """Convert line spectral frequencies to prediction filter coefficients
        returns a vector a containing the prediction filter coefficients from a vector lsf of line spectral frequencies.

        """
        #   Reference: A.M. Kondoz, "Digital Speech: Coding for Low Bit Rate Communications
        #   Systems" John Wiley & Sons 1994 ,Chapter 4

        # Line spectral frequencies must be real.

        lsf = np.array(lsf)

        if max(lsf) > np.pi or min(lsf) < 0:
            raise ValueError('Line spectral frequencies must be between 0 and pi.')

        p = len(lsf) # model order

        # Form zeros using the LSFs and unit amplitudes
        z  = np.exp(1.j * lsf)

        # Separate the zeros to those belonging to P and Q
        rQ = z[0::2]
        rP = z[1::2]

        # Include the conjugates as well
        rQ = np.concatenate((rQ, rQ.conjugate()))
        rP = np.concatenate((rP, rP.conjugate()))

        # Form the polynomials P and Q, note that these should be real
        Q  = np.poly(rQ);
        P  = np.poly(rP);

        # Form the sum and difference filters by including known roots at z = 1 and
        # z = -1

        if p%2:
            # Odd order: z = +1 and z = -1 are roots of the difference filter, P1(z)
            P1 = sp.convolve(P, [1, 0, -1])
            Q1 = Q
        else:
            # Even order: z = -1 is a root of the sum filter, Q1(z) and z = 1 is a
            # root of the difference filter, P1(z)
            P1 = sp.convolve(P, [1, -1])
            Q1 = sp.convolve(Q, [1,  1])

        # Prediction polynomial is formed by averaging P1 and Q1

        a = .5 * (P1+Q1)
        return a[0:-1:1] # do not return last element


    def poly2lsf(a):
        """Prediction polynomial to line spectral frequencies.

        converts the prediction polynomial specified by A,
        into the corresponding line spectral frequencies, LSF.
        normalizes the prediction polynomial by A(1).

        """

        #Line spectral frequencies are not defined for complex polynomials.

        # Normalize the polynomial

        a = np.array(a)
        if a[0] != 1:
            a/=a[0]

        if max(np.abs(np.roots(a))) >= 1.0:
            raise ValueError('The polynomial must have all roots inside of the unit circle.');


        # Form the sum and difference filters

        p  = len(a)-1   # The leading one in the polynomial is not used
        a1 = np.concatenate((a, np.array([0])))
        a2 = a1[-1::-1]
        P1 = a1 - a2        # Difference filter
        Q1 = a1 + a2        # Sum Filter

        # If order is even, remove the known root at z = 1 for P1 and z = -1 for Q1
        # If odd, remove both the roots from P1

        if p%2: # Odd order
            P, r = sp.deconvolve(P1,[1, 0 ,-1])
            Q = Q1
        else:          # Even order
            P, r = sp.deconvolve(P1, [1, -1])
            Q, r = sp.deconvolve(Q1, [1,  1])

        rP  = np.roots(P)
        rQ  = np.roots(Q)

        aP  = np.angle(rP[1::2])
        aQ  = np.angle(rQ[1::2])

        lsf = sorted(np.concatenate((-aP,-aQ)))

        return lsf



    @property
    def mid_buffer_size(self):
        return self._mid_buffer_size

    @property
    def buffer(self):
        return self._buffer

    @property
    def pitch_rate(self):
        print("pitch_rate = {}".format(self._pitch_rate))
        return self._pitch_rate

    @pitch_rate.setter
    def pitch_rate(self, new_pitch_rate):
        self._pitch_rate = new_pitch_rate
        self.initialize(self._pitch_rate, self.frame_format)




