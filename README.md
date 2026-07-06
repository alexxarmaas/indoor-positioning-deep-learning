# Indoor Positioning Campaign Manager - Deep Learning Extension

GPU-accelerated deep learning and ensemble methods for BLE fingerprint-based indoor positioning.

> [!IMPORTANT]
> This repository is derived from [SimonRusu/IPSCampaignManager](https://github.com/SimonRusu/IPSCampaignManager), originally created by **Simon Rusu** as his Bachelor's Thesis. The original Git history is preserved. The deep-learning extension and the campaign aggregation, experimentation, and visualization work were developed by [@juanfradelrm](https://github.com/juanfradelrm).

## Extension overview

This version extends the original indoor positioning platform with:

- A scikit-learn MLP ensemble for RSSI fingerprint regression.
- A GPU-accelerated PyTorch ResMLP ensemble with automatic CUDA/CPU selection.
- A hybrid scikit-learn and PyTorch positioning method across BLE channels 37, 38, and 39.
- Multi-day campaign aggregation and normalized reference-point imports.
- Reproducible training, calibration, validation, and benchmark experiments.
- Comparative accuracy and precision charts for Deep Learning and WKNN results.
- Backend unit tests for positioning, campaign aggregation, and data processing.

## Original project

### Data Processing System for Indoor Positioning Improvement
#### Integration with a Bluetooth Low Energy Data Collection Robotic Platform

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

## Deep Learning positioning

The DeepLearning method is an ensemble of multilayer perceptrons for RSSI fingerprint regression. It uses the same source signals, aleatory evaluation points, prediction storage, and 3D Euclidean error metric as WKNN.

- Training data: one fingerprint per reference capture whose rotation matches the aleatory evaluation, plus a strongest-signal aggregate repeated six times to balance daily variation with the evaluation distribution.
- Hidden layers: 64 and 32 neurons with Tanh activation.
- Optimizer: Adam with early stopping for full campaigns; L-BFGS remains the fallback for very small datasets.
- Preprocessing: missing RSSI values become `-110 dBm`; inputs and target coordinates are standardized independently.
- Reproducibility: deterministic seed 45 and L2 regularization alpha 0.01.
- Dependencies: no new package is required; the implementation uses scikit-learn already listed in `IPSCM/requirements.txt`.

On the aggregated application campaign, using iBeacon, channel 37, unlimited samples, and the configured 0-degree aleatory captures, the model obtained 1.705 m mean error, 1.354 m median, and 2.738 m P90. The best WKNN result across k=1..20 was 2.025 m, 1.543 m, and 2.962 m respectively (k=11, Manhattan distance). Deep Learning reduced these three errors by 15.8%, 12.2%, and 7.6%. This is a campaign-specific benchmark rather than a guarantee for every protocol/channel combination.

The SQLite files in `Campañas/` are source datasets and are not discovered automatically at startup. A single import now accepts one aleatory database followed by any number of reference databases. The reference databases are normalized to the common point map from the `.conf` file and stored in one visible campaign, so measurements from every collection day contribute to the same fingerprint. Upload the files in this order:

1. Aleatory SQLite database.
2. All reference SQLite databases, in acquisition order.
3. General BLE configuration, aleatory point configuration, and reference point configuration.

The campaign database is stored in the `ipscm-data` Docker volume and therefore survives container rebuilds. Choose **Sin límite** in the sample selector when every RSSI sample from every imported day must be considered.

To run the application and compare both methods:

```bash
docker compose up --build
```

Open `http://localhost:4200`, apply `DeepLearning` and `WKNN` with the same protocol, channel, and RSSI sample limit, then use **Generar gráficas** on the aggregated campaign. Both results are stored in `Method_Prediction`. The precision view plots error CDFs and the accuracy view plots predicted positions; both views also show mean, median, P90, and maximum-error tables.

Run the backend method tests with:

```bash
docker compose build server
docker compose run --rm server python -m unittest discover -s tests -v
```

---
## Authors and attribution

- **Original project and Bachelor's Thesis:** Simon Rusu - [LinkedIn](https://www.linkedin.com/in/simon-rusu-5069aa280) | [GitHub](https://github.com/SimonRusu)
- **Deep-learning extension:** [@juanfradelrm](https://github.com/juanfradelrm)

## License status

The upstream repository did not include a root license file when this derivative was created. Copyright remains with the respective authors; contact them before reusing or redistributing the code beyond the permissions granted by GitHub's Terms of Service.
