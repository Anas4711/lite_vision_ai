# 📦 lite_vision_ai — Flutter TensorFlow Lite Image Classification Library

# 🧠 LiteVision AI

A simple, lightweight Flutter library for TensorFlow Lite image classification.  
Load your `.tflite` model, classify images, and retrieve predictions in just a few lines of code!

---

## 🚀 Features

✅ Load any TensorFlow Lite model  
✅ Classify local images directly from device storage  
✅ Retrieve top predictions with confidence scores  
✅ Works seamlessly on Android, iOS, and desktop (with proper setup)  
✅ Minimal and developer-friendly API design  

---

## 🧩 Example Usage

 import 'dart:io';
import 'package:lite_vision_ai/lite_vision_ai.dart';

void main() async {
  final vision = LiteVisionAI();
  await vision.load(
    model: 'assets/models/model.tflite',
    labels: 'assets/models/labels.txt',
  );
  await vision.classify(image: File('assets/test_image.jpg'), top: 3);
  print('🧠 Top Label: ${vision.name}');
  print('🎯 Confidence: ${vision.accuracy.toStringAsFixed(2)}%');
  print('📊 All Predictions: ${vision.predictions}');
}

## 📦 Installation

```yaml
dependencies:
  lite_vision_ai: ^1.0.0
  
---

