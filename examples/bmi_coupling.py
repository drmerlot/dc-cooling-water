"""Drive the model through its Basic Model Interface the way a coupling framework would: give it each hour's air
temperature and humidity, step it, and read the water fluxes. Here the 'other model' is a simple synthetic weather
generator; in practice it would be an atmosphere or land-surface component."""
import math
import tempfile

import numpy as np

from dc_cooling_water import CoolingWaterBmi

config = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
config.write("design: hybrid\nit_capacity_mw: 300\nend_time: 604800\n")   # one week, hourly steps
config.close()

model = CoolingWaterBmi()
model.initialize(config.name)
temp, rh = np.empty(1), np.empty(1)
withdrawn = np.empty(1)
total_m3 = 0.0
while model.get_current_time() < model.get_end_time():
    hour = model.get_current_time() / 3600
    temp[0] = 27 + 6 * math.sin(2 * math.pi * (hour % 24 - 9) / 24)      # a hot week
    rh[0] = 55 - 15 * math.sin(2 * math.pi * (hour % 24 - 9) / 24)
    model.set_value("atmosphere_bottom_air__temperature", temp)
    model.set_value("atmosphere_bottom_air_water_vapor__relative_saturation", rh)
    total_m3 += model.get_value("data_center_cooling_water__withdrawal_volume_flux", withdrawn)[0] * model.get_time_step()
    model.update()
model.finalize()
print(f"Water withdrawn over the week: {total_m3:,.0f} m3 ({total_m3 * 264.172 / 7:,.0f} gallons a day)")
