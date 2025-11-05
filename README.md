# 🌄 Brighten a Dark Cave Image using Gamma Correction

## 📘 Project Description
This project enhances the brightness and visibility of a **dark cave image** using **gamma correction**.  
Gamma correction is a nonlinear transformation used to adjust image brightness, making dark regions more visible without overexposing brighter areas.

---

## 🎯 Objective
To **brighten a dark cave image** by applying gamma correction using **Python and OpenCV**.

---

## ⚙️ How It Works
1. Load the input image using OpenCV.  
2. Apply the gamma correction formula:

   \[
   \text{Output Pixel} = 255 \times \left(\frac{\text{Input Pixel}}{255}\right)^\gamma
   \]

3. For brightening, choose **γ < 1** (e.g., 0.4–0.6).  
4. Display and save the enhanced image.

---

## 🧠 What is Gamma Correction?
Gamma correction compensates for the nonlinear way our eyes perceive light.  
- **γ < 1** → Brightens the image  
- **γ > 1** → Darkens the image  

This technique is widely used in photography, graphics, and image preprocessing.

---

## 🚀 Steps to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
