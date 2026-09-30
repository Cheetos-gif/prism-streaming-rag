# Industrial Compressor Emergency Shutdown Protocols and Recovery Workflows

## §1 Immediate Emergency Shutdown Triggers
Plant operators and maintenance personnel must immediately execute an emergency manual shutdown of any Model 7 or Model 9 compressor upon observing any of the following four critical triggers:
1. **Visible Smoke**: Any smoke or vapor plume issuing from the electric drive motor, rotary air-end casing, or electrical cabinet.
2. **Persistent Burning Smell**: Pungent odor of overheated synthetic lubricant, burning electrical insulation, or friction-damaged drive couplings.
3. **Metal-on-Metal Sound**: High-pitched acoustic screeching, severe rhythmic grinding, or catastrophic knocking indicating rotor-stator contact or total bearing collapse.
4. **Discharge Pressure Exceeding 150 PSI**: Delivery line or vessel pressure spiking **above 150 PSI** (10.34 bar). Since Model 7 nominal operating pressure is 125 PSI and safety relief is calibrated at 140 PSI, pressure above 150 PSI represents an acute over-pressurization hazard requiring instant tripping.

## §2 Immediate Emergency Shutdown Procedure
Upon encountering any trigger, immediately execute the following steps:
1. **Hit Emergency Stop**: Strike the red mushroom-head Emergency Stop (E-Stop) pushbutton on the front control panel.
2. **Electrical Isolation (LOTO)**: Switch the 400V main feeder breaker to "OFF" and attach a physical lockout padlock.
3. **Evacuation**: Evacuate non-essential personnel outside the compressor room (maintain a 5-meter boundary).
4. **Fire Suppression**: If open combustion occurs, use Class ABC dry-chemical or CO2 fire extinguishers. Never apply water to energized electrical gear or hot oil sumps.

## §3 Mandatory Cooldown and Depressurization Procedure
Strict cooldown protocols must precede any physical intervention:
- **Minimum 15-Minute Cooldown Period**: The compressor enclosure and access panels must remain closed for a **minimum of 15 minutes** following emergency shutdown. Opening panels earlier exposes personnel to superheated pressurized oil mists and thermal shock risks.
- **Zero Pressure Verification**: After the 15-minute cooldown, verify on analog gauges that receiver and separator vessel pressures have fallen to 0.0 PSI before unlatching doors or loosening fittings.

## §4 Post-Shutdown Restart Verification Checklist
No unit that underwent an emergency shutdown may be restarted until an engineer signs off on this checklist:
- [ ] Root cause of the emergency trip identified, documented, and repaired.
- [ ] Manual shaft rotation: Rotor turned freely by hand through 3 full revolutions (360°) via drive coupling without binding or scraping noises.
- [ ] Oil inspection: Fluid level verified in sight glass; sample inspected for thermal degradation, foaming, or particulate debris.
- [ ] Fasteners checked: Foundation bolts and motor mounts torqued to 110 Nm.
- [ ] Electrical check: Motor insulation resistance (Megger test) verified > 50 MΩ at 1,000V.

## §5 Incident Reporting Requirements
- **Initial Notification**: File a digital incident report in the PRISM Safety Portal within **4 hours** of the emergency shutdown.
- **Comprehensive Report**: Submit a root-cause investigation report with telemetry logs and component photos to the Plant Safety Director within **24 hours**.
