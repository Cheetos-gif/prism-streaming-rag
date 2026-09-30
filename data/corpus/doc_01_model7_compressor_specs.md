# Model 7 Rotary Screw Compressor Technical Specifications

## §1 General Description and Engineering Overview
The Model 7 Rotary Screw Industrial Compressor is a high-efficiency, single-stage oil-injected rotary screw air compressor engineered for heavy-duty industrial manufacturing environments. Manufactured under PRISM Industrial Machinery specifications, the Model 7 is purpose-built for continuous-duty applications requiring continuous volumetric efficiency, low vibration output, and automated closed-loop telemetry monitoring. The unit integrates an asymmetric rotary screw profile with direct-drive mechanical coupling, minimizing transmission losses and ensuring uniform volumetric delivery across varied operational cycles.

## §2 Operating Ratings and Performance Specifications
The operational performance envelope of the Model 7 compressor is defined by the following certified factory ratings:
- **Rated Volumetric Output**: 150 CFM (Cubic Feet per Minute) / 4.25 m³/min at nominal operating pressure.
- **Nominal Discharge Working Pressure**: 125 PSI (Pounds per Square Inch) / 8.62 bar.
- **Maximum Allowable Working Pressure (MAWP)**: 135 PSI (9.31 bar) with safety valve relief calibrated to 140 PSI.
- **Drive Motor Power**: 30 kW (40 HP) 3-phase squirrel-cage induction motor, operating at 400V / 50 Hz (or 460V / 60 Hz optional configuration) with an operating speed of 2,950 RPM.
- **Motor Enclosure & Insulation**: Totally Enclosed Fan Cooled (TEFC) with IP55 environmental ingress protection rating and Class F thermal insulation.
- **Operating Ambient Temperature Range**: -10°C to 45°C (14°F to 113°F). Continuous operation beyond 45°C triggers automated thermal derating.
- **Acoustic Noise Emission Level**: 78 dB(A) measured at 1.0 meter distance under full loaded operation, enclosed within the standard acoustic canopy.

## §3 Physical Dimensions and Weight Specifications
The physical architecture of the Model 7 compressor is designed for compact industrial footprint and floor mounting:
- **Total Operating Weight**: 340 kg (dry shipping weight: 328 kg).
- **External Enclosure Dimensions**: 1,450 mm (Length) × 850 mm (Width) × 1,220 mm (Height).
- **Mounting Configuration**: Four-point rigid sub-base isolated with elastomeric anti-vibration damping pads, secured using M16 foundation anchoring bolts.
- **Pneumatic Discharge Outlet**: 1.0-inch NPT female threaded pipe connection.
- **Condensate Drain Connection**: 1/4-inch BSP automatic electronic drain port.

## §4 Lubrication and Fluid Systems
The internal fluid circuit serves the dual purpose of rotor sealing, cooling, and mechanical bearing lubrication:
- **Lubricant Reservoir Capacity**: 12.0 Liters total system volume.
- **Recommended Lubricant Type**: Synthetic PAO ISO 46 (Polyalphaolefin synthetic rotary compressor fluid). Use of mineral oils or non-certified lubricants is strictly prohibited.
- **Lubricant Carryover Rate**: Less than 3 ppm (parts per million) downstream of the three-stage air-oil coalescing separator.
- **Thermal Regulation**: Integrated thermostatic mixing bypass valve opening at 71°C to direct hot oil through the aluminum air-cooled bar-and-plate heat exchanger.

## §5 Control Instrumentation and Telemetry Interface
Operational telemetry is processed by the onboard PRISM SmartControl-X controller:
- **Pressure Transducer Accuracy**: ±0.5 PSI across the full 0–175 PSI operational range.
- **Discharge Temperature Sensor**: Class A PT100 RTD sensor calibrated from -20°C to 150°C.
- **Telemetry Communications**: RS-485 Modbus RTU and CANbus interfaces streaming real-time motor current, discharge pressure, oil temperature, and cumulative operating hours to local plant SCADA and remote live RAG monitoring agents.
