# ITC5 Radar Stabilization

## Overview
Cyber-physical platform that stabilizes a close-range inspection radar by driving orthogonal stepper stages from vision feedback. A Teensy 4.1 executes either an LQR regulator ([Control/lqr_control.ino](Control/lqr_control.ino)) or a dual-axis proportional controller ([Control/p_control.ino](Control/p_control.ino)), while a Raspberry Pi runs motion-estimation ROS 2 nodes (optical flow, template matching, ECC) and ToF-based scale compensation.

## Hardware
- Teensy 4.1 with TB6600 stepper drivers (belt axis X, dual-rod axis Y) and AS5600 encoders.
- Raspberry Pi HQ camera streaming YUV420 via `rpicam-vid`.
- VL53L5CX ToF sensor for dynamic mm/px scaling ([Sensor reading/tof_distance_node.py](Sensor reading/tof_distance_node.py)).
- MPU6050 IMU (DMP mode) for belt acceleration logging.

## Software Stack
| Layer | Components |
| --- | --- |
| Firmware | [`StepperAxisX`/`StepperAxisY` ISR engines, IMU logger, SD CSV writer inside lqr_control/p_control](Control/lqr_control.ino) |
| ROS 2 nodes | Camera publisher ([Motion estimation/camera_node.py](Motion estimation/camera_node.py)), LK flow ([Motion estimation/pyr_lk_node.py](Motion estimation/pyr_lk_node.py)), ToF publisher, optional ECC & template matching nodes |
| Messaging | UART link Teensy↔Pi (`T,<t>` tags, vision feedback lines), ROS topics `/vision/*`, `/sensors/tof/distance_mm` |

## Repository Layout
```
Control/                Teensy sketches (P)
Motion estimation/      Active ROS2 nodes (camera relay, LK OF)
Sensor reading/         VL53L5CX ToF publisher
Out of scope/           Additional motion-estimation experiments
sysID/                  Reserved for plant identification data
```

## Calibration & Tuning
- **Camera scale**: `CAM_MM_SCALE`, `CAM_MM_SCALE_Y` in [Control/p_control.ino](Control/p_control.ino).  
- **Setpoints**: `REF_X_MM`, `REF_Y_MM` in both control sketches.  
- **Gains**: `KP_CAM_X`, `KP_CAM_Y' derived from MATLAB model and fine-tuned on the prototype.  
- **ToF fusion**: `tof_reference_distance_mm`, `tof_alpha` in LK node.


## Testing
<p align="center">
<table>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/80d4a136-49f1-4757-8954-b2ec4134439d"
             width="300"
             controls>
      </video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/80249cf1-4727-4d93-a1a0-96a379f2892a"
             width="300"
             controls>
      </video>
    </td>
  </tr>
</table>
</p>


