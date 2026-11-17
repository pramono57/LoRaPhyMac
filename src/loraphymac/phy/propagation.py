# ================================================================
# phy/propagation.py
# Physical DATA_DOWN / DATA_DOWN_FWD + 
# Relay APC RX cleanup, APC TX index, and timing validation
# ITM  
# ekuatorial Indonesia  Equatorial EN0 = 360.0 radio_climate = 1
# ================================================================

import math

import numpy as np
from loraphymac.pyitm.itm import ITMAreadBLoss

class ITM:

    def __init__(self, EPS=25.0, SGM=0.02, EN0=360.0,
                 radio_climate=1, pol=1, pctTime=0.7,
                 pctLoc=0.5, pctConf=0.90):
        
        # Baseline ekuatorial Indonesia  Equatorial EN0 = 360.0 radio_climate = 1
        # pctTime = 0.5   # placeholder; unused when ModVar=0
        # pctLoc  = 0.5   # placeholder; unused when ModVar=0
        # pctConf = 0.9   # 90% combined variability quantile
        
        self.EPS = EPS
        self.SGM = SGM
        self.EN0 = EN0
        self.radio_climate = radio_climate
        self.pol = pol
        self.pctTime = pctTime
        self.pctLoc = pctLoc
        self.pctConf = pctConf
        """
        from  pyitm
        ModVar (int): 0 - Single: pctConf is "Time/Situation/Location",
                        pctTime, pctLoc not used
                      1 - Individual: pctTime is "Situation/Location",
                        pctConf is "Confidence", pctLoc not used
                      2 - Mobile: pctTime is "Time/Locations (Reliability)",
                        pctConf is "Confidence", pctLoc not used
                      3 - Broadcast: pctTime is "Time", pctLoc is "Location",
                        pctConf is "Confidence"
        deltaH (float): Terrain irregularity parameter [m].  This is the
            interdecile range of terrain elevation between Tx/Rx sites.  For
            average terrain use 90.  Other recommendations are:
            Flat (or smooth water): 0; Plains: 30; Hills: 90; Mountains:
            200; Rugged Mountains: 500
        tht_m (float): Transmitter antenna height [m].  The height is
            determined by the center of the radiating element above ground.
            [0.5 - 3000]
        rht_m (float): Receiver antenna height [m].  The height is
            determined by the center of the radiating element above ground.
            [0.5 - 3000]
        dist_km (float): Distance between antennas [km]. The model is valid
            for distances in the range 1km - 2000km.
        TSiteCriteria (int): Tx Antenna deployment sitting criteria:
            0 - random, 1 - careful, 2 - very careful
        RSiteCriteria (int): Rx Antenna deployment sitting criteria:
            0 - random, 1 - careful, 2 - very careful
        eps_dielect (float): Relative Permittivity of the earth
        sgm_conductivity (float): Conductivity of the earth [S/m]
        eno_ns_surfref (float): Surface Refractivity [250 - 400 N-units]
        frq_mhz (float): Carrier frequency [MHz]; The model is valid for
            frequencies in the range 20MHz - 20GHz
        radio_climate (int): 1-Equatorial, 2-Continental Subtropical,
                             3-Maritime Tropical, 4-Desert, 5-Continental
                             Temperate, 6-Maritime Temperate, Over Land,
                             7-Maritime Temperate, Over Sea
        pol (int): Antenna polarization; 0-Horizontal, 1-Vertical.  It is
            assumed that both antenna elements have the same polarization.
        pctTime (float): Time Reliability Percentage [.01 to .99]
        pctLoc (float): Location Reliability Percentage [.01 to .99]
        pctConf (float): Confidence Interval Percentage [.01 to .99]
        """
    


    def pathloss(self, d_m, freq_mhz, h_tx, h_rx, terrain_delta_h=90):
        d_km = d_m / 1000
        # ITM requires minimum distance of 1 km
        d_km = max(d_km, 1.0)
        
        pl = ITMAreadBLoss(
            0, terrain_delta_h, h_tx, h_rx, d_km,
            1, 0,
            self.EPS, self.SGM, self.EN0,
            freq_mhz, self.radio_climate, self.pol,
            self.pctTime, self.pctLoc, self.pctConf
        )
        return float(pl)

   # def compute_pathloss(self, d_m, freq_mhz, h_tx, h_rx, terrain_delta_h=90):
   #     return self.pathloss(d_m, freq_mhz, h_tx, h_rx, terrain_delta_h)


class FSPL:
    """
    Free Space Path Loss (FSPL)
    """

    def pathloss(self, d_m, freq_mhz, h_tx=None, h_rx=None, deltaH=None,**kwargs):
        d_km = d_m / 1000.0
        pl = 32.45 + 20 * np.log10(d_km) + 20 * np.log10(freq_mhz)
        return pl


class IEEE80211ah:
    """IEEE 802.11ah Outdoor Path Loss Model.

    Based on Equation (15) of Islam et al. (2023),
    Performance Evaluation of Multi-Hop LoRaWAN.

    Distance: meters
    Frequency: MHz
    Path loss: dB

    This model is distance-based and does not incorporate terrain
    irregularity (``terrain_delta_h``).
    """

    def pathloss(
        self,
        d_m,
        freq_mhz,
        h_tx=None,
        h_rx=None,
        terrain_delta_h=None,
        **kwargs,
    ):
        distance_m = float(d_m)
        frequency_mhz = float(freq_mhz)
        if not math.isfinite(distance_m) or distance_m <= 0:
            raise ValueError("distance_m must be finite and greater than zero")
        if not math.isfinite(frequency_mhz) or frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be finite and greater than zero")

        path_loss_db = (
            23.3
            + 37.6 * math.log10(distance_m)
            + 21.0 * math.log10(frequency_mhz / 900.0)
        )
        if not math.isfinite(path_loss_db):
            raise ValueError("path loss must be finite")
        return path_loss_db


class OkumuraHata:
    def __init__(self, environment="urban"):
        self.environment = environment.lower()

    @staticmethod
    def correction_term_a_hm(freq_mhz, h_m):
        return (1.1*np.log10(freq_mhz)-0.7)*h_m - (1.56*np.log10(freq_mhz)-0.8)

    def pathloss(self, d_m, freq_mhz, h_tx, h_rx, terrain_delta_h=None, **kwargs):
        d_km = d_m / 1000.0
        a_hm = OkumuraHata.correction_term_a_hm(freq_mhz, h_rx)

        # Hata Urban
        pl_urban = (
            69.55 + 26.16*np.log10(freq_mhz)
            - 13.82*np.log10(h_tx)
            - a_hm
            + (44.9 - 6.55*np.log10(h_tx))*np.log10(d_km)
        )

        if self.environment == "suburban":
            return pl_urban - 2*(np.log10(freq_mhz/28)**2) - 5.4

        if self.environment == "rural":
            return pl_urban - 4.78*(np.log10(freq_mhz)**2) \
                             + 18.33*np.log10(freq_mhz) \
                             - 40.94

        return pl_urban
