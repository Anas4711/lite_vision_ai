# 📸 lite_vision_ai

[![Pub Version](https://img.shields.io/pub/v/lite_vision_ai.svg)](https://pub.dev/packages/lite_vision_ai)
[![GitHub Stars](https://img.shields.io/github/stars/Anas4711/lite_vision_ai.svg)](https://github.com/Anas4711/lite_vision_ai/stargazers)
[![License](https://img.shields.io/github/license/Anas4711/lite_vision_ai.svg)](https://opensource.org/licenses/MIT)
![Platform Support](https://img.shields.io/badge/platform-android%20|%20ios-blue)
![Null Safety](https://img.shields.io/badge/null%20safety-supported-success)

A lightweight, fast, and elegant Flutter library for on-device **image classification** using **TensorFlow Lite**.  
Designed to be **simple**, **offline-first**, and **easy to integrate** into production apps with minimal boilerplate.

---

## ✨ Features

- ⚡ **100% Offline:** Runs completely on-device with zero internet dependency.
- 🚀 **High Performance:** Fast image inference powered by `tflite_flutter`.
- 🎯 **Custom Models:** Easily load your own `.tflite` models and `.txt` labels.
- 📊 **Top-N Predictions:** Configurable top prediction confidence percentages.
- 🌐 **Cross-Platform:** Supports Android and iOS.
- 🛠️ **Flutter 3 & Image v4+ Compatible:** Built-in modern pixel normalization and safe byte buffers.

---

## 📦 Installation

Add `lite_vision_ai` to your `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  lite_vision_ai: ^1.0.3
```
Then run:

```bash
flutter pub get
```

## ⚙️ Setup & Assets
1.	Create an assets/models/ folder in your project root and add your .tflite model and .txt labels file.
2.	Declare them in your pubspec.yaml:

```yaml
flutter:
  uses-material-design: true
  assets:
    - assets/models/model.tflite
    - assets/models/labels.txt
  ```

  💡 Note: Ensure your labels.txt file lists one class label per line.

## 🚀 Quick Start
Here is a complete example showing how to initialize, load, and classify an image:

```dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:lite_vision_ai/lite_vision_ai.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 1. Instantiate the class
  final vision = LiteVisionAI();

  // 2. Load your TFLite model & labels
  await vision.load(
    model: 'assets/models/model.tflite',
    labels: 'assets/models/labels.txt',
  );

  // 3. Classify an image file
  final imageFile = File('path/to/your/image.jpg');
  await vision.classify(image: imageFile, top: 3);

  // 4. Access prediction results
  if (vision.isReady) {
    print('🏷️ Top Label: ${vision.name}');
    print('🎯 Accuracy: ${vision.accuracy.toStringAsFixed(2)}%');
    print('📊 Top Predictions: ${vision.predictions}');
  }
}
```

## 📖 API Reference

| Member | Type | Description |
| :--- | :--- | :--- |
| `load({required model, required labels})` | `Future<void>` | Loads `.tflite` model and `.txt` labels from app assets. |
| `classify({required image, int top = 3})` | `Future<void>` | Runs image inference and extracts the top N predicted labels. |
| `name` | `String` | Returns the label of the top-1 prediction. |
| `accuracy` | `double` | Returns the confidence percentage (%) of the top-1 prediction. |
| `predictions` | `Map<String, double>` | Returns top-N labels mapped to their confidence percentages. |
| `isReady` | `bool` | Returns `true` if the model has been loaded successfully. |


## 💡 Example Project

Check out the [`example/`](./example) directory for a fully working Flutter application demonstrating real-time image picking and classification.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Anas4711/lite_vision_ai/blob/main/LICENSE) file for details.
