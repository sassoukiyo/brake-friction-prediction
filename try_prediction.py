

import numpy as np
from helpers import load_model

model = load_model("XX_000.pkl")

rotor_speed_rpm      = 100.0
brake_pressure_bar   = 2.5
normal_force_kN      = 80.0
pad_temperature_C    = 90.0
rotor_temperature_C  = 110.0

my_input = np.array([[
    rotor_speed_rpm,
    brake_pressure_bar,
    normal_force_kN,
    pad_temperature_C,
    rotor_temperature_C,
]])

prediction = model.predict(my_input)

print("Your input:")
print(f"  rotor_speed_rpm      = {rotor_speed_rpm}")
print(f"  brake_pressure_bar   = {brake_pressure_bar}")
print(f"  normal_force_kN      = {normal_force_kN}")
print(f"  pad_temperature_C    = {pad_temperature_C}")
print(f"  rotor_temperature_C  = {rotor_temperature_C}")
print()
print(f"Predicted friction_coefficient: {prediction[0]:.4f}")
