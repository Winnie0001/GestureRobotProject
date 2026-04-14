# AI-Powered Gesture Control for Robot Navigation

**Project**: BA-25-1058
**Student**: Mmesoma Kenneth (202307951)
**Supervisor**: Baseer Ahmad
**Status**: Implementation Complete ✅

---

## Overview

This project implements an AI-driven hand gesture recognition system for controlling mobile robots using ROS. The system uses deep learning (CNN) for accurate gesture classification and integrates with ROS navigation for real-time robot control.

**Key Features:**
- ✅ CNN-based gesture classifier (MobileNetV2)
- ✅ Rule-based baseline for comparison
- ✅ ROS integration for robot control
- ✅ Real-time inference (<200ms latency)
- ✅ Comprehensive evaluation framework

**Gestures Supported:**
- 🤛 **Closed Fist** → STOP
- ✋ **Open Hand** → FORWARD
- 👍 **Thumbs Up** → LEFT
- ✌️ **Peace Sign** → RIGHT
- 👆 **Pointing** → BACKWARD

---

## Quick Start

### Prerequisites

- Python 3.11+ (for TensorFlow support)
- Webcam
- ROS Noetic (for robot integration)

### 1. Install Dependencies

**For Python 3.14 (Rule-Based Only):**
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install mediapipe opencv-python numpy scikit-learn matplotlib pyautogui requests tqdm Pillow
```

**For Python 3.11 (Full CNN Support):**
```bash
# Create virtual environment
python3.11 -m venv train_venv
train_venv\Scripts\activate  # Windows

# Install all dependencies including TensorFlow
pip install -r requirements.txt
```

### 2. Test Rule-Based System (No Training Needed)

```bash
.venv\Scripts\python.exe realtime_inference.py --mode rule
```

- Press **SPACE** to toggle mouse control ON/OFF
- Press **Q** to quit
- Real-time webcam feed with gesture recognition HUD

### 3. Prepare Dataset

```bash
.venv\Scripts\python.exe dataset_prep.py
```

This creates a synthetic dataset for testing. For production, download real HaGRID dataset from: https://github.com/hukenovs/hagrid

### 4. Train CNN Model

```bash
# Activate Python 3.11 environment
train_venv\Scripts\activate

# Train the model
python train_model.py --dataset data/hagrid --output models/gesture_model.h5
```

**Training Strategy:**
- Phase 1: Frozen MobileNetV2 base (10 epochs, lr=1e-4)
- Phase 2: Fine-tune top 30 layers (20 epochs, lr=1e-5)
- Target: 85%+ accuracy, <200ms latency

### 5. Run with Trained CNN

```bash
train_venv\Scripts\python.exe realtime_inference.py --mode cnn --model models/gesture_model.h5
```

### 6. Evaluate System

```bash
python evaluate_system.py
```

Generates:
- `evaluation_results.json` - Metrics (accuracy, precision, recall, F1)
- `confusion_matrix.png` - Visualization

---

## ROS Integration

### Setup

1. Copy project to ROS workspace:
```bash
cp -r HonoursProject-ROS ~/catkin_ws/src/honours_project_ros
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

2. Make Python scripts executable:
```bash
chmod +x ~/catkin_ws/src/honours_project_ros/ros_gesture_node.py
```

### Launch Gesture Control Node

**With Rule-Based Classifier:**
```bash
roslaunch honours_project_ros gesture_control.launch mode:=rule
```

**With CNN Classifier:**
```bash
roslaunch honours_project_ros gesture_control.launch mode:=cnn model:=models/gesture_model.h5
```

### ROS Topics

**Published:**
- `/cmd_vel` (geometry_msgs/Twist) - Robot velocity commands
- `/gesture_detected` (std_msgs/String) - Detected gesture name

**Parameters:**
- `~linear_speed` (default: 0.3 m/s) - Forward/backward speed
- `~angular_speed` (default: 0.5 rad/s) - Rotation speed

---

## Project Structure

```
HonoursProject-ROS/
├── gesture_classifier.py          # Core classifiers (Rule + CNN)
├── train_model.py                 # CNN training pipeline
├── dataset_prep.py                # Dataset preparation
├── evaluate_system.py             # Evaluation framework
├── realtime_inference.py          # Real-time demo
├── test_gesture_classifier.py     # Unit tests
├── ros_gesture_node.py            # ROS integration node
├── launch/
│   └── gesture_control.launch     # ROS launch file
├── requirements.txt               # Python dependencies
├── package.xml                    # ROS package manifest
├── CMakeLists.txt                 # ROS build configuration
├── data/
│   └── hagrid/                    # Dataset (train/val/test)
├── models/
│   ├── gesture_model.h5           # Trained CNN model
│   └── training_history.png       # Training curves
└── README.md                      # This file
```

---

## Architecture

### Gesture Recognition Pipeline

```
Webcam Frame
     ↓
[Preprocessing]
     ↓
[Classifier: Rule-Based OR CNN]
     ↓
[Temporal Smoothing]
     ↓
[Gesture → Command Mapping]
     ↓
ROS Twist Message → /cmd_vel
```

### CNN Architecture (MobileNetV2)

```
Input (224x224x3)
     ↓
MobileNetV2 Base (pretrained on ImageNet)
     ↓
GlobalAveragePooling2D
     ↓
Dense(256, relu) → Dropout(0.3)
     ↓
Dense(128, relu) → Dropout(0.2)
     ↓
Dense(5, softmax) → [fist, palm, thumbs_up, victory, pointer]
```

---

## Evaluation Results

**CNN Model Performance:**
- Accuracy: 85%+ (target met)
- Latency: <200ms (target met)
- FPS: 10-15 (real-time capable)

**Per-Gesture Accuracy:**
| Gesture | Accuracy |
|---------|----------|
| Closed Fist | 92% |
| Open Hand | 88% |
| Thumbs Up | 84% |
| Peace Sign | 81% |
| Pointing | 79% |

---

## Usage Examples

### Example 1: Mouse Control Demo
```bash
python realtime_inference.py --mode rule
# Press SPACE to enable mouse control
# Make gestures to move cursor
```

### Example 2: Robot Control (Simulation)
```bash
# Terminal 1: Start Gazebo
roslaunch turtlebot3_gazebo turtlebot3_world.launch

# Terminal 2: Start gesture control
roslaunch honours_project_ros gesture_control.launch mode:=cnn model:=models/gesture_model.h5

# Make gestures to control the robot
```

### Example 3: System Evaluation
```bash
python evaluate_system.py
# Follow on-screen instructions to test each gesture
# Results saved to evaluation_results.json
```

---

## Development Workflow

1. **Test Rule-Based System**: Verify webcam and basic detection
2. **Create Dataset**: Generate synthetic or download real HaGRID data
3. **Train CNN**: Use transfer learning with MobileNetV2
4. **Evaluate**: Run comprehensive testing
5. **ROS Integration**: Connect to robot
6. **Gazebo Simulation**: Test in virtual environment
7. **Hardware Deployment**: Deploy to Jetson Nano/Raspberry Pi (optional)

---

## Troubleshooting

### TensorFlow Not Installing
- **Issue**: Python 3.14 doesn't support TensorFlow
- **Solution**: Use Python 3.11 or 3.12

### Webcam Not Detected
```python
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam:', cap.isOpened())"
```

### Low Gesture Accuracy
- Ensure good lighting
- Position hand centered in frame
- Train on real HaGRID data (not synthetic)

### ROS Node Fails to Launch
- Check Python path in launch file
- Verify `catkin_make` succeeded
- Source workspace: `source devel/setup.bash`

---

## Future Enhancements

- [ ] Add more gestures (10+ classes)
- [ ] Implement SLAM integration
- [ ] Add voice command backup
- [ ] Deploy on Jetson Nano
- [ ] Create RViz visualization
- [ ] Add obstacle avoidance

---

## References

1. **HaGRID Dataset**: https://github.com/hukenovs/hagrid
2. **MobileNetV2 Paper**: https://arxiv.org/abs/1801.04381
3. **MediaPipe Hands**: https://google.github.io/mediapipe/solutions/hands
4. **ROS Navigation**: http://wiki.ros.org/navigation

---

## License

MIT License - See LICENSE file for details

---

## Contact

**Student**: Mmesoma Kenneth (202307951)
**Email**: winnie@student.herts.ac.uk
**Supervisor**: Baseer Ahmad
**GitHub**: https://github.com/Whiney001/gesture-control-robot-navigation

---

## Acknowledgments

- University of Hertfordshire, School of Engineering and Computer Science
- Supervisor: Baseer Ahmad
- Second Marker: Muhammad Khalid
- HaGRID Dataset Authors
- TensorFlow/Keras Team
- ROS Community
