######################################################
#
# Script to convert between different units of gas flux
# 
# User inputs a single value or a file containing a list of values
# then chooses a unit to convert from and a unit to convert to.
#
# The code uses a series of functions to convert any input unit to L/min
# Then converts from L/min to the chosen output unit. This is to allow for
# easier addition of different units as only two functions need to be added
# rather than many different functions.
#
# Requires:
#  * numpy
#
######################################################

"""
Unit IDs
1.0 L/min \n
1.1 L/s \n
1.2 L/hr \n
1.3 L/day \n
1.4 L/yr \n
1.5 cm3/s \n
 \n
2.0 g/min \n
2.1 ton/day \n
2.2 ton/year \n
2.3 kg/day \n
2.4 kg/year \n
2.5 kg/min \n
 \n
3.0 mol/min \n
3.1 mol/s \n

"""

#~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~#

def pressure(depth):
    """
    Converts pressure to depth by using 1 atm increase per 10m depth.
    Depth (m), pressure (Pa)
    """
    return 101325+10132.5*depth

def getMolarMass(gas):
    if gas=="co2" or gas=="CO2":
        molarMass = 44.01
    elif gas=="ch4" or gas=="CH4":
        molarMass = 16.04
    else:
        raise ValueError("Input gas must be CH4 or CO2")
    return molarMass

def massPL(depth,gas):
    """
    Produces the mass of one letre of gas at a specific depth
    vol (L), depth (m), mass (g)
    """
    molarMass = getMolarMass(gas)
    P = pressure(depth)/101325
    n = (P*1)/(0.0821*298)
    print("Pressure: ",P)
    print("Density (g/L): ",n*molarMass)
    return n*molarMass 

def mass_moles(mass,gas):
    """
    Produces number of moles from a mas of a given gas
    """
    molarMass = getMolarMass(gas)
    return mass/molarMass


#~~~~~~~~~~~~~~~~~~~~~~ Mass / Time ~~~~~~~~~~~~~~~~~~~~~~~~#
# Standard: grams / min

#L/min --> g/min
def LMin_gMin(LMin,depth,gas):
    massPerLetre = massPL(depth,gas)
    return LMin*massPerLetre
#g/min --> L/min
def gMin_LMin(gMin,depth,gas):
    massPerLetre = massPL(depth,gas)
    return gMin/massPerLetre

#Same dimension conversion

#g/min --> ton/day
def gMin_tonDay(gMin):
    return gMin*(60*24)/(1000*1000)
#ton/day --> g/min
def tonDay_gMin(tonDay):
    return tonDay*(1000*1000)/(60*24)

#g/min --> ton/year
def gMin_tonYear(gMin):
    return gMin*(60*24*365)/(1000*1000)
#ton/year --> g/min
def tonYear_gMin(tonYear):
    return tonYear*(1000*1000)/(60*24*365)

#g/min --> kg/day
def gMin_kgDay(gMin):
    return gMin*(1440/1000)
#kg/day --> g/min
def kgDay_gMin(kgDay):
    return kgDay*(1000/1440)

#g/min --> kg/year
def gMin_kgYear(gMin):
    return gMin*(525960/1000)
#kg/year --> g/min
def kgYear_gMin(kgYear):
    return kgYear*(1000/525960)

#g/min --> kg/min
def gMin_kgMin(gMin):
    return gMin/1000
#kg/min --> g/min
def kgMin_gMin(kgMin):
    return kgMin*1000


#~~~~~~~~~~~~~~~~~~~~~~~~ Moles / Time ~~~~~~~~~~~~~~~~~~~~~~#
# Standard: moles / min

#L/min --> mol/min
def LMin_molMin(LMin,depth,gas):
    P = pressure(depth)
    return (P*LMin*0.001)/(8.314*298)
#mol/min --> L/min
def molMin_LMin(molMin,depth,gas):
    P = pressure(depth)
    return ((molMin*8.314*298)/P)*1000

#Same dimension conversion

#mol/s --> mol/min
def molS_molMin(molS):
    return molS/60
#mol/min --> mol/s
def molMin_molS(molMin):
    return molMin*60


#~~~~~~~~~~~~~~~~~~~~~~~ Volume / Time ~~~~~~~~~~~~~~~~~~~~~~~#
# Standard: Litres / min

#Same dimension converions

#L/min --> cm3/s
def LMin_cm3S(LMin):
    return LMin*(1000/60)
#cm3/s --> L/min
def cm3S_LMin(cm3S):
    return cm3S*(60/1000)

#L/min --> L/s
def LMin_LS(LMin):
    return LMin/60
#L/s --> L/min
def LS_LMin(LS):
    return LS*60

#L/min --> L/hr
def LMin_LHr(LMin):
    return LMin*60
#L/hr --> L/min
def LHr_LMin(LHr):
    return LHr/60

#L/min --> L/day
def LMin_LDay(LMin):
    return LMin*1440
#L/day --> L/min
def LDay_LMin(LDay):
    return LDay/1440

#L/min --> L/year
def LMin_LYr(LMin):
    return LMin*525600
#L/year --> L/min
def LYr_LMin(LYr):
    return LYr/525600




    

################################################################
#                            GUI                               #
################################################################

def convert_units(input_value, from_unit, to_unit, gas, depth):
    """
    Convert a value from one unit to another.
    Returns the converted value.
    """
    try:
        value = float(input_value)
    except ValueError:
        return "Invalid input"
    
    # Convert input to L/min (standard)
    if from_unit == "L/min":
        lmin_value = value
    elif from_unit == "L/s":
        lmin_value = LS_LMin(value)
    elif from_unit == "L/hr":
        lmin_value = LHr_LMin(value)
    elif from_unit == "L/day":
        lmin_value = LDay_LMin(value)
    elif from_unit == "L/yr":
        lmin_value = LYr_LMin(value)
    elif from_unit == "cm3/s":
        lmin_value = cm3S_LMin(value)
    elif from_unit == "g/min":
        lmin_value = gMin_LMin(value, depth, gas)
    elif from_unit == "ton/day":
        gmin_value = tonDay_gMin(value)
        lmin_value = gMin_LMin(gmin_value, depth, gas)
    elif from_unit == "ton/year":
        gmin_value = tonYear_gMin(value)
        lmin_value = gMin_LMin(gmin_value, depth, gas)
    elif from_unit == "kg/day":
        gmin_value = kgDay_gMin(value)
        lmin_value = gMin_LMin(gmin_value, depth, gas)
    elif from_unit == "kg/year":
        gmin_value = kgYear_gMin(value)
        lmin_value = gMin_LMin(gmin_value, depth, gas)
    elif from_unit == "kg/min":
        gmin_value = kgMin_gMin(value)
        lmin_value = gMin_LMin(gmin_value, depth, gas)
    elif from_unit == "mol/min":
        lmin_value = molMin_LMin(value, depth, gas)
    elif from_unit == "mol/s":
        molmin_value = molS_molMin(value)
        lmin_value = molMin_LMin(molmin_value, depth, gas)
    else:
        return "Unknown input unit"
    
    # Convert from L/min to output unit
    if to_unit == "L/min":
        result = lmin_value
    elif to_unit == "L/s":
        result = LMin_LS(lmin_value)
    elif to_unit == "L/hr":
        result = LMin_LHr(lmin_value)
    elif to_unit == "L/day":
        result = LMin_LDay(lmin_value)
    elif to_unit == "L/yr":
        result = LMin_LYr(lmin_value)
    elif to_unit == "cm3/s":
        result = LMin_cm3S(lmin_value)
    elif to_unit == "g/min":
        result = LMin_gMin(lmin_value, depth, gas)
    elif to_unit == "ton/day":
        gmin_value = LMin_gMin(lmin_value, depth, gas)
        result = gMin_tonDay(gmin_value)
    elif to_unit == "ton/year":
        gmin_value = LMin_gMin(lmin_value, depth, gas)
        result = gMin_tonYear(gmin_value)
    elif to_unit == "kg/day":
        gmin_value = LMin_gMin(lmin_value, depth, gas)
        result = gMin_kgDay(gmin_value)
    elif to_unit == "kg/year":
        gmin_value = LMin_gMin(lmin_value, depth, gas)
        result = gMin_kgYear(gmin_value)
    elif to_unit == "kg/min":
        gmin_value = LMin_gMin(lmin_value, depth, gas)
        result = gMin_kgMin(gmin_value)
    elif to_unit == "mol/min":
        result = LMin_molMin(lmin_value, depth, gas)
    elif to_unit == "mol/s":
        molmin_value = LMin_molMin(lmin_value, depth, gas)
        result = molMin_molS(molmin_value)
    else:
        return "Unknown output unit"
    
    return result


################################################################
#                            MAIN                              #
################################################################

if __name__=="__main__":
    import tkinter as tk
    from tkinter import ttk
    
    # Create main window
    root = tk.Tk()
    root.title("Gas Emission Rate Converter")
    root.geometry("500x400")
    root.resizable(False, False)
    
    # Define units
    units = [
        "L/min", "L/s", "L/hr", "L/day", "L/yr", "cm3/s",
        "g/min", "ton/day", "ton/year", "kg/day", "kg/year", "kg/min",
        "mol/min", "mol/s"
    ]
    
    gases = ["CO2", "CH4"]
    
    # Create and configure frames
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Title
    title_label = ttk.Label(main_frame, text="Gas Emission Rate Converter", 
                           font=("Arial", 16, "bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    # Input value
    ttk.Label(main_frame, text="Input Value:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, pady=5)
    input_entry = ttk.Entry(main_frame, width=30)
    input_entry.grid(row=1, column=1, pady=5, padx=10)
    
    # From unit
    ttk.Label(main_frame, text="From Unit:", font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, pady=5)
    from_unit_var = tk.StringVar(value=units[0])
    from_unit_combo = ttk.Combobox(main_frame, textvariable=from_unit_var, 
                                   values=units, state="readonly", width=27)
    from_unit_combo.grid(row=2, column=1, pady=5, padx=10)
    
    # To unit
    ttk.Label(main_frame, text="To Unit:", font=("Arial", 10)).grid(row=3, column=0, sticky=tk.W, pady=5)
    to_unit_var = tk.StringVar(value=units[1])
    to_unit_combo = ttk.Combobox(main_frame, textvariable=to_unit_var, 
                                 values=units, state="readonly", width=27)
    to_unit_combo.grid(row=3, column=1, pady=5, padx=10)
    
    # Gas type
    ttk.Label(main_frame, text="Gas Type:", font=("Arial", 10)).grid(row=4, column=0, sticky=tk.W, pady=5)
    gas_var = tk.StringVar(value=gases[0])
    gas_combo = ttk.Combobox(main_frame, textvariable=gas_var, 
                            values=gases, state="readonly", width=27)
    gas_combo.grid(row=4, column=1, pady=5, padx=10)
    
    # Depth
    ttk.Label(main_frame, text="Depth (m):", font=("Arial", 10)).grid(row=5, column=0, sticky=tk.W, pady=5)
    depth_entry = ttk.Entry(main_frame, width=30)
    depth_entry.insert(0, "0")
    depth_entry.grid(row=5, column=1, pady=5, padx=10)
    
    # Result label
    result_var = tk.StringVar(value="")
    result_label = ttk.Label(main_frame, textvariable=result_var, 
                            font=("Arial", 12, "bold"), foreground="blue")
    result_label.grid(row=7, column=0, columnspan=2, pady=20)
    
    # Convert function
    def perform_conversion():
        try:
            input_val = input_entry.get()
            from_u = from_unit_var.get()
            to_u = to_unit_var.get()
            gas = gas_var.get()
            depth = float(depth_entry.get())
            
            result = convert_units(input_val, from_u, to_u, gas, depth)
            
            if isinstance(result, str):
                result_var.set(result)
            else:
                result_var.set(f"Result: {result:.6e} {to_u}")
        except Exception as e:
            result_var.set(f"Error: {str(e)}")
    
    # Convert button
    convert_button = ttk.Button(main_frame, text="Convert", command=perform_conversion)
    convert_button.grid(row=6, column=0, columnspan=2, pady=15)
    
    # Bind Enter key to perform conversion
    root.bind('<Return>', lambda event: perform_conversion())
    
    # Start GUI
    root.mainloop()

        
























    
        
