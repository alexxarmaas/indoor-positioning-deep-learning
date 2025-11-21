# Data Processing System for Indoor Positioning Improvement  
### Integration with a Bluetooth Low Energy Data Collection Robotic Platform

This repository contains the official documentation of my **Bachelor’s Thesis (TFG)**, focused on enhancing indoor positioning accuracy through automated data collection using a robotics-based platform. The project integrates Bluetooth Low Energy (BLE) signal acquisition with multiple positioning algorithms inside a modular and extensible processing system.

---

## 📄 Project Documentation

### **1. Project Proposal**  
[PropuestaTFG_SimonRusu.pdf](https://github.com/user-attachments/files/23674447/PropuestaTFG_SimonRusu.pdf)

Contains the full project description extracted above.




### **2. Final Thesis Document**  
[TFG_Simon_Rusu_v2_AQ_GdB.pdf](https://github.com/user-attachments/files/23674449/TFG_Simon_Rusu_v2_AQ_GdB.pdf)

Complete and final version of the thesis, including theoretical background, implementation details, software architecture, algorithms’ evaluation, experiments, results, and conclusions.

## 🧠 Project Overview

Indoor positioning systems (IPS) face significant accuracy challenges due to signal variability, interference, and environmental dynamics. Traditional fingerprinting-based approaches require extensive manual data collection and calibration.

This project proposes a **robotics-driven, fully automated approach** to improve IPS calibration through:

- Autonomous collection of BLE and Wi-Fi signal strength values (RSSI)
- Integration with a proprietary database used by the robotic platform
- Evaluation of multiple IPS algorithms, including:  
  - **Triangulation**  
  - **Centroid-based methods**  
  - **RSSI-based techniques**  
  - **Hybrid positioning strategies**
- Comparative visualisation using specialised charts to analyse the deviation between estimated and real positions
- A **modular software architecture**, allowing easy incorporation and testing of new positioning methodologies in the future  

---

## 🛠️ Technologies & Tools

The project relies on a selected stack of mature, industry-proven technologies to ensure robustness, scalability and seamless integration with the robotic data-collection platform:

### **Docker**
Used to containerize services, standardize environments, and ensure consistent deployment across machines through lightweight, isolated containers.

### **Python**
Backend implementation built in Python to maintain full compatibility with the existing Robomap ecosystem.  
Core components include:  
- **Flask** for the REST API  
- **SQLAlchemy** for ORM-based data access  
- **PyPI** for dependency management

### **JavaScript / TypeScript**
Frontend developed using modern web technologies to deliver a responsive, component-based interface.  
Key technologies:  
- **Angular** for application structure and UI components  
- **Node.js** for tooling and package management  
- **Bulma** and complementary libraries for interface styling and interaction

### **SQLite**
Lightweight relational database chosen for interoperability with Robomap, efficient handling of large data-collection campaigns, and seamless integration with Python and SQLAlchemy.

### **Visual Studio Code**
Primary development environment, providing an optimized workflow with integrated debugging, extension support and Git-based version control.

### **Git & GitHub**
Version control and repository hosting used to maintain code integrity, track changes and support collaborative workflows.

---

## 🚀 Key Contributions

- Automated alternative to manual fingerprinting calibration  
- Unified framework for testing multiple indoor positioning algorithms  
- Scalable design suitable for future research and extensions  
- Real-world evaluation using a robotic data collection platform  

---

## 👤 Author

**Simon Rusu**  
🔗 LinkedIn: [www.linkedin.com/in/simon-rusu-5069aa280](https://www.linkedin.com/in/simon-rusu-5069aa280)

