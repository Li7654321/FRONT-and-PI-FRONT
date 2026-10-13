#!/usr/bin/env python
"""
Differentiable waveform synthesis from Green's functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Compute the source time function (STF) spectrum and phase delay.
@torch.jit.script
def compute_stf_complex(omega: torch.Tensor, half_rise: torch.Tensor, rupt_time: torch.Tensor) -> torch.Tensor:
    arg = omega * half_rise

    safe_arg = torch.where(arg == 0.0, torch.ones_like(arg), arg)

    # Avoid division by zero in the sinc term.
    sinc_term = torch.where(arg == 0.0, torch.ones_like(arg), torch.sin(safe_arg) / safe_arg)
    sinc_sq = sinc_term ** 2

    # Include rupture onset and half-rise-time delays.
    phase_delay = half_rise + rupt_time
    phase_angle = -1.0 * omega * phase_delay

    real_part = sinc_sq * torch.cos(phase_angle)
    imag_part = sinc_sq * torch.sin(phase_angle)

    return torch.complex(real_part, imag_part)


class WaveformSynthesizer(nn.Module):
    def __init__(self, gf_shape, dt, max_vals, gf_tensor=None):
        super(WaveformSynthesizer, self).__init__()
        self.gf_shape = gf_shape
        self.dt = dt
        self.scales = max_vals

        self.nt = gf_shape[3]

        # Double the FFT length to reduce circular-convolution wraparound.
        self.pad_nt = self.nt * 2
        self.total_bins = self.pad_nt // 2 + 1

        nyquist = 1.0 / (2.0 * dt)
        cutoff_freq = 0.5  # Hz

        self.valid_bins = int((cutoff_freq / nyquist) * (self.pad_nt // 2)) + 1

        if gf_tensor is not None:
            if not isinstance(gf_tensor, torch.Tensor):
                gf_tensor = torch.tensor(gf_tensor, dtype=torch.float32)

            # Zero-pad the Green's functions before the FFT.
            gf_padded = F.pad(gf_tensor, (0, self.pad_nt - self.nt))

            gf_spec_val = torch.fft.rfft(gf_padded, dim=-1)
            gf_spec_truncated = gf_spec_val[..., :self.valid_bins]
            self.register_buffer('gf_spec_trunc', gf_spec_truncated)
        else:
            self.gf_spec_trunc = None

    def forward(self, src, rup):
        batch_size = src.size(0)

        # Restore physical units from normalized source parameters.
        ss_slip = src[:, 0:1, :] * self.scales['slip']
        ds_slip = src[:, 1:2, :] * self.scales['slip']

        # Keep the rise time positive for a well-defined STF.
        rise_time = src[:, 2:3, :] * self.scales['rise'] + 1e-2
        rupt_time = rup * self.scales['rupt']

        freqs = torch.fft.rfftfreq(self.pad_nt, d=self.dt, device=src.device)[:self.valid_bins]
        omega = 2.0 * torch.pi * freqs
        omega_grid = omega.view(1, 1, 1, -1)

        half_rise_exp = (rise_time * 0.5).permute(0, 2, 1).unsqueeze(-1)
        rupt_time_exp = rupt_time.permute(0, 2, 1).unsqueeze(-1)

        # Apply the delayed STF in the frequency domain.
        stf_spec_complex = compute_stf_complex(omega_grid, half_rise_exp, rupt_time_exp)

        ss_c = ss_slip.permute(0, 2, 1).unsqueeze(-1).to(torch.complex64)
        ds_c = ds_slip.permute(0, 2, 1).unsqueeze(-1).to(torch.complex64)

        gf_ss = self.gf_spec_trunc[:, 0, :, :].unsqueeze(0)
        gf_ds = self.gf_spec_trunc[:, 1, :, :].unsqueeze(0)

        # Sum the strike-slip and dip-slip responses over all subfaults.
        base_resp = ss_c * gf_ss + ds_c * gf_ds
        full_resp_sub = base_resp * stf_spec_complex
        total_spec_trunc = torch.sum(full_resp_sub, dim=1)

        # Restore the full FFT spectrum with zeroed high-frequency bins.
        pad_size = self.total_bins - self.valid_bins
        total_spec = F.pad(total_spec_trunc, (0, pad_size))

        # Transform back to time and discard the padding.
        waveforms_padded = torch.fft.irfft(total_spec, n=self.pad_nt, dim=-1)
        waveforms = waveforms_padded[..., :self.nt]

        # Return (batch, time, station, component).
        waveforms = waveforms.permute(0, 2, 1)
        n_sta = self.gf_shape[2] // 3
        waveforms = waveforms.view(batch_size, self.nt, n_sta, 3)

        return waveforms
