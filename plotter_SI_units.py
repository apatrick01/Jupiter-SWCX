# -*- coding: utf-8 -*-
"""
Allows for the different graphs to be easily plotted without needing to change the code
"""
from run_functions_SI_units import *

menu = """Q - Quit
1 - Magnetosphere Boundaries
2 - Neutral H Density
3 - Proton Density
4 - O8+ Ion Density
5 - C6+ Ion Density
6 - Proton Temperature
7 - Thermal Velocity
8 - Bulk Velocity
9 - Relative Velocity
10 - O8+ Emission Rate
11 - Plot Line of Sight 1
12 - Plot Line of Sight 2
13 - Plot Line of Sight 3
14 - Plot Line of Sight 1 with O8+ Emission Rate
15 - Plot Line of Sight 2 with O8+ Emission Rate
16 - Plot Line of Sight 3 with O8+ Emission Rate
17 - Plot all 3 Lines of Sight with O8+ Emission Rate
18 - Calculate Intensity along Line of Sight 1
19 - Calculate Intensity along Line of Sight 2
20 - Calculate Intensity along Line of Sight 3
21 - Plot Logarithmic O8+ Emission rate
22 - Plot Intensity of Magnetosheath Slice
"""

select = "a"

print(menu)

while select != "q" or select != "Q":
    select = input("Enter a number between 1 and 22: ")
    
    if select == "1":
        msphere_bounds()
    elif select == "2":
        neut_dens()
    elif select == "3":
        pro_dens()
    elif select == "4":
        O8_ion_dens()
    elif select == "5":
        C6_ion_dens()
    elif select == "6":
        temp()
    elif select == "7":
        thermal_velocity()
    elif select == "8":
        bulk_velocity()
    elif select == "9":
        relative_velocity()
    elif select == "10":
        O8_emission()
    elif select == "11":
        los_1()
    elif select == "12":
        los_2()
    elif select == "13":
        los_3()
    elif select == "14":
        los_1_emission()
    elif select == "15":
        los_2_emission()
    elif select == "16":
        los_3_emission()
    elif select == "17":
        all_los_emission()
    elif select == "18":
        los_1_int()
    elif select == "19":
        los_2_int()
    elif select == "20":
        los_3_int()
    elif select == "21":
        O8_emission_log()
    elif select == "22":
        int_bar()
    elif select == "q" or select == "Q":
        print("Quitting")
        break
    else:
        print("That option is invalid")