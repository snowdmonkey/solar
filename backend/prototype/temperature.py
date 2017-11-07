from math import exp, sqrt, log


class TempTransformer:

    def __init__(self, e=1, od=1, rtemp=20, atemp=None, irwtemp=None,
                 irt=1, rh=50, pr1=21106.77, pb=1501.0, pf=1, po=-7340, pr2=0.012545258):
        """
        the class aim to transfer the raw sensor data in flir infrared image to celsius degree
        :param e: Emissivity, should be ~0.95 to 0.97, defaults to 1
        :param od: object distance in meters
        :param rtemp: apparent reflected temperature
        :param atemp: atmospheric temperature for transmission loss, defaults to rtemp
        :param irwtemp: infrared window temperature, defaults to rtemp
        :param irt: infrared window transmission, defaults to 1, should be ~0.95 to 0.96
        :param rh: relative humidity in percentage, defaults to 50%
        :param pr1: planck R1 calibration constant
        :param pb: planck B calibration constant
        :param pf: planck F calibration constant
        :param po: planck O calibration constant
        :param pr2: planck R2 calibration constant
        """
        self.e = e
        self.od = od
        self.rtemp = rtemp
        if atemp is None:
            self.atemp = rtemp
        else:
            self.atemp = atemp
        if irwtemp is None:
            self.irwtemp = rtemp
        else:
            self.irwtemp = irwtemp
        self.irt = irt
        self.rh = rh
        self.pr1 = pr1
        self.pb = pb
        self.pf = pf
        self.po = po
        self.pr2 = pr2

    def raw2temp(self, raw: int) -> float:
        ata1 = 0.006569  # atmospheric trans alpha 1
        ata2 = 0.01262  # atmospheric trans alpha 2
        atb1 = -0.002276  # atmospheric trans beta 1
        atb2 = -0.00667  # atmospheric trans beta 2
        atx = 1.9  # atmospheric trans x

        emiss_wind = 1-self.irt
        refl_wind = 0
        h2o = (self.rh / 100) * \
            exp(1.5587 + 0.06939 * self.atemp - 0.00027816 * self.atemp**2 + 0.00000068455 * self.atemp**3)
        tau1 = atx * exp(-1 * sqrt(self.od/2) * (ata1 + atb1 * sqrt(h2o))) + \
            (1 - atx) * exp(-1 * sqrt(self.od / 2) * (ata2 + atb2 * sqrt(h2o)))
        tau2 = atx * exp(-1 * sqrt(self.od / 2) * (ata1 + atb1 * sqrt(h2o))) + \
            (1 - atx) * exp(-1 * sqrt(self.od / 2) * (ata2 + atb2 * sqrt(h2o)))

        raw_refl1 = self.pr1 / (self.pr2 * (exp(self.pb / (self.rtemp + 273.15)) - self.pf)) - self.po
        raw_refl1_attn = (1 - self.e) / self.e * raw_refl1
        raw_atm1 = self.pr1 / (self.pr2 * (exp(self.pb / (self.atemp + 273.15)) - self.pf)) - self.po
        raw_atm1_attn = (1 - tau1) / self.e / tau1 * raw_atm1
        raw_wind = self.pr1 / (self.pr2 * (exp(self.pb / (self.irwtemp + 273.15)) - self.pf)) - self.po
        raw_wind_attn = emiss_wind / self.e / tau1 / self.irt * raw_wind
        raw_refl2 = self.pr1 / (self.pr2 * (exp(self.pb / (self.rtemp + 273.15)) - self.pf)) - self.po
        raw_refl2_attn = refl_wind / self.e / tau1 / self.irt * raw_refl2
        raw_atm2 = self.pr1 / (self.pr2 * (exp(self.pb / (self.atemp + 273.15)) - self.pf)) - self.po
        raw_atm2_attn = (1 - tau2) / self.e / tau1 / self.irt / tau2 * raw_atm2
        raw_obj = raw / self.e / tau1 / self.irt / tau2 - \
            raw_atm1_attn - raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn
        temp_c = self.pb / log(self.pr1 / (self.pr2 * (raw_obj + self.po)) + self.pf) - 273.15
        return temp_c
