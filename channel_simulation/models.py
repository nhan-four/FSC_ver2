"""Channel models used by the semantic communication simulators.

This project historically referenced `channel_simulation.models.*` but the
implementation was missing in the repo snapshot. This module provides a small,
consistent API across channels:

- transmit(tx_symbols) -> (rx_symbols, channel_info)
- set_snr(snr_db)
- equalize(rx_symbols, method='zf'|'mmse') for fading channels

Notes
-----
The rest of the codebase uses BPSK by default (real-valued +/-1 symbols), but we
implement channels in complex baseband so the same models work for QPSK (complex)
if used elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _snr_db_to_linear(snr_db: float) -> float:
    return 10 ** (snr_db / 10.0)


def _complex_awgn(shape, noise_var: float) -> np.ndarray:
    """Complex AWGN with total variance noise_var per complex sample."""
    sigma = np.sqrt(noise_var / 2.0)
    return sigma * (np.random.randn(*shape) + 1j * np.random.randn(*shape))


@dataclass
class AWGNChannel:
    """Additive White Gaussian Noise channel."""

    snr_db: float = 10.0

    def set_snr(self, snr_db: float) -> None:
        self.snr_db = float(snr_db)

    @property
    def noise_var(self) -> float:
        # Assume unit average symbol energy.
        snr_lin = _snr_db_to_linear(self.snr_db)
        return 1.0 / snr_lin

    def transmit(self, tx_symbols: np.ndarray) -> Tuple[np.ndarray, Dict]:
        tx = np.asarray(tx_symbols)
        n = _complex_awgn(tx.shape, self.noise_var)
        rx = tx.astype(np.complex128) + n
        info = {
            "channel": "awgn",
            "snr_db": float(self.snr_db),
            "noise_var": float(self.noise_var),
        }
        return rx, info


@dataclass
class RayleighChannel:
    """Flat Rayleigh fading channel with complex Gaussian coefficient."""

    snr_db: float = 10.0

    # Per-call fading coefficient is stored for equalization.
    h: np.ndarray | None = None

    def set_snr(self, snr_db: float) -> None:
        self.snr_db = float(snr_db)

    @property
    def noise_var(self) -> float:
        snr_lin = _snr_db_to_linear(self.snr_db)
        return 1.0 / snr_lin

    def _sample_fading(self, n: int) -> np.ndarray:
        # CN(0,1): E|h|^2 = 1
        return (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2.0)

    def transmit(self, tx_symbols: np.ndarray) -> Tuple[np.ndarray, Dict]:
        tx = np.asarray(tx_symbols).astype(np.complex128)
        self.h = self._sample_fading(tx.size).reshape(tx.shape)
        n = _complex_awgn(tx.shape, self.noise_var)
        rx = self.h * tx + n
        info = {
            "channel": "rayleigh",
            "snr_db": float(self.snr_db),
            "noise_var": float(self.noise_var),
        }
        return rx, info

    def equalize(self, rx_symbols: np.ndarray, method: str = "mmse") -> np.ndarray:
        if self.h is None:
            return rx_symbols
        y = np.asarray(rx_symbols).astype(np.complex128)
        h = self.h
        if method.lower() == "zf":
            return y / (h + 1e-12)
        # MMSE
        nv = self.noise_var
        return y * np.conj(h) / (np.abs(h) ** 2 + nv)


@dataclass
class RicianChannel:
    """Flat Rician fading channel (LOS + scattered)."""

    snr_db: float = 10.0
    k_factor_db: float = 3.0  # K-factor in dB

    h: np.ndarray | None = None

    def set_snr(self, snr_db: float) -> None:
        self.snr_db = float(snr_db)

    @property
    def noise_var(self) -> float:
        snr_lin = _snr_db_to_linear(self.snr_db)
        return 1.0 / snr_lin

    def _sample_fading(self, n: int) -> np.ndarray:
        k_lin = _snr_db_to_linear(self.k_factor_db)
        # Deterministic LOS component
        h_los = np.ones(n, dtype=np.complex128)
        # Scattered component CN(0,1)
        h_scatter = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2.0)
        return np.sqrt(k_lin / (k_lin + 1.0)) * h_los + np.sqrt(1.0 / (k_lin + 1.0)) * h_scatter

    def transmit(self, tx_symbols: np.ndarray) -> Tuple[np.ndarray, Dict]:
        tx = np.asarray(tx_symbols).astype(np.complex128)
        self.h = self._sample_fading(tx.size).reshape(tx.shape)
        n = _complex_awgn(tx.shape, self.noise_var)
        rx = self.h * tx + n
        info = {
            "channel": "rician",
            "snr_db": float(self.snr_db),
            "k_factor_db": float(self.k_factor_db),
            "noise_var": float(self.noise_var),
        }
        return rx, info

    def equalize(self, rx_symbols: np.ndarray, method: str = "mmse") -> np.ndarray:
        if self.h is None:
            return rx_symbols
        y = np.asarray(rx_symbols).astype(np.complex128)
        h = self.h
        if method.lower() == "zf":
            return y / (h + 1e-12)
        nv = self.noise_var
        return y * np.conj(h) / (np.abs(h) ** 2 + nv)


class LoRaChannel:
    """Simplified LoRa/LoRaWAN-like channel model.

    This channel is controlled purely by SNR (consistent with AWGN/Rayleigh/Rician).
    It models LoRa-like characteristics with:
    - Flat Rayleigh small-scale fading (optional, mild)
    - AWGN noise controlled by snr_db
    
    The `snr_db` parameter directly controls the noise level, ensuring fair
    comparison with other channels in SNR sweep experiments.
    
    Any additional kwargs (distance, sf, tx_power) are ignored for backward
    compatibility but do not affect the channel behavior.
    """

    def __init__(self, snr_db: float = 10.0, **kwargs):
        """Initialize LoRaChannel.
        
        Parameters:
        -----------
        snr_db : float
            Signal-to-noise ratio in dB. This directly controls noise variance.
        **kwargs
            Ignored for backward compatibility (distance, sf, tx_power, etc.)
        """
        self.snr_db = float(snr_db)
        self.h = None

    def set_snr(self, snr_db: float) -> None:
        """Update SNR setting."""
        self.snr_db = float(snr_db)

    @property
    def noise_var(self) -> float:
        """Compute noise variance from SNR.
        
        Assumes unit average symbol energy, consistent with other channels.
        """
        snr_lin = _snr_db_to_linear(self.snr_db)
        return 1.0 / snr_lin

    def _sample_fading(self, n: int) -> np.ndarray:
        """Sample Rayleigh small-scale fading CN(0,1).
        
        This provides mild multipath fading typical of LoRa channels.
        The fading is normalized so E|h|^2 = 1.
        """
        return (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2.0)

    def transmit(self, tx_symbols: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Transmit symbols through LoRa channel.
        
        Parameters:
        -----------
        tx_symbols : np.ndarray
            Complex baseband transmit symbols
            
        Returns:
        --------
        rx_symbols : np.ndarray
            Received symbols after channel
        info : dict
            Channel information including snr_db and noise_var
        """
        tx = np.asarray(tx_symbols).astype(np.complex128)
        
        # Apply mild Rayleigh fading (typical for LoRa multipath)
        self.h = self._sample_fading(tx.size).reshape(tx.shape)
        
        # Generate AWGN noise based on SNR
        n = _complex_awgn(tx.shape, self.noise_var)
        
        # Received signal: h * tx + n
        rx = self.h * tx + n
        
        info = {
            "channel": "lora",
            "snr_db": float(self.snr_db),
            "noise_var": float(self.noise_var),
        }
        return rx, info

    def equalize(self, rx_symbols: np.ndarray, method: str = "mmse") -> np.ndarray:
        """Equalize received symbols to compensate for fading.
        
        Parameters:
        -----------
        rx_symbols : np.ndarray
            Received symbols after channel
        method : str
            Equalization method: 'zf' (zero-forcing) or 'mmse' (minimum mean-square error)
            
        Returns:
        --------
        equalized : np.ndarray
            Equalized symbols
        """
        if self.h is None:
            return rx_symbols
        y = np.asarray(rx_symbols).astype(np.complex128)
        h = self.h
        if method.lower() == "zf":
            return y / (h + 1e-12)
        # MMSE equalization
        nv = self.noise_var
        return y * np.conj(h) / (np.abs(h) ** 2 + nv)
