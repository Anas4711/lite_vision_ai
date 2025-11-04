library lite_vision_ai;

import 'dart:io';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

/// ---------------------------------------------------------------------------
/// 📘 LiteVisionAI
/// A simple, elegant Flutter library for TensorFlow Lite image classification.
/// Designed for developers who want to easily load, analyze, and interpret
/// deep learning models directly in their apps.
/// ---------------------------------------------------------------------------

/// Represents the result of an image classification operation.
///
/// Contains:
/// - [name] → The top predicted class label.
/// - [accuracy] → Confidence of the top prediction.
/// - [predictions] → A list of top results (class name → confidence %).
class PredictionResult {
  final String name;
  final double accuracy;
  final Map<String, double> predictions;

  PredictionResult({
    required this.name,
    required this.accuracy,
    required this.predictions,
  });
}

/// ---------------------------------------------------------------------------
/// 🧠 LiteVisionAI
/// Core class that handles loading, running, and reading results
/// from TensorFlow Lite image classification models.
/// ---------------------------------------------------------------------------
class LiteVisionAI {
  late Interpreter _interpreter;
  late List<String> _labels;
  bool _ready = false;

  String _name = 'No analysis performed yet';
  double _accuracy = 0.0;
  Map<String, double> _predictions = const {};

  /// Loads a TensorFlow Lite model (.tflite) and label file (.txt).
  ///
  /// Example:
  /// ```dart
  /// final vision = LiteVisionAI();
  /// await vision.load(
  ///   model: 'assets/models/model.tflite',
  ///   labels: 'assets/models/labels.txt',
  /// );
  /// ```
  Future<void> load({
    required String model,
    required String labels,
  }) async {
    final modelData = await rootBundle.load(model);
    _interpreter = Interpreter.fromBuffer(modelData.buffer.asUint8List());

    final labelData = await rootBundle.loadString(labels);
    _labels = labelData.split('\n').where((e) => e.isNotEmpty).toList();

    _ready = true;
  }

  /// Runs image classification on a given [image] file.
  ///
  /// - [image]: The image file to analyze.
  /// - [top]: The number of top predictions to return (default = 3).
  ///
  /// Example:
  /// ```dart
  /// await vision.classify(image: File('example.jpg'), top: 5);
  /// print(vision.name);
  /// print(vision.accuracy);
  /// print(vision.predictions);
  /// ```
  Future<void> classify({
    required File image,
    int top = 3,
  }) async {
    if (!_ready) throw Exception('⚠️ Model not loaded. Call load() first.');

    // Decode image bytes
    final bytes = await image.readAsBytes();
    final img.Image? original = img.decodeImage(bytes);
    if (original == null) throw Exception('❌ Failed to decode image.');

    // Resize to 224x224 (standard for most TFLite models)
    const inputSize = 224;
    final resized = img.copyResize(original, width: inputSize, height: inputSize);

    // Normalize pixel values to [-1, 1]
    final input = List.generate(1, (_) {
      return List.generate(inputSize, (y) {
        return List.generate(inputSize, (x) {
          final p = resized.getPixel(x, y);
          return [
            (p.r - 127.5) / 127.5,
            (p.g - 127.5) / 127.5,
            (p.b - 127.5) / 127.5,
          ];
        });
      });
    });

    // Run model inference
    final output = List.filled(_labels.length, 0.0).reshape([1, _labels.length]);
    _interpreter.run(input, output);
    final scores = output[0];

    // Sort predictions (highest confidence first)
    final ranked = List.generate(scores.length, (i) => MapEntry(i, scores[i]))
      ..sort((a, b) => b.value.compareTo(a.value));

    // Keep only top N results
    final Map<String, double> topResults = {
      for (var i = 0; i < top && i < ranked.length; i++)
        _labels[ranked[i].key]: (ranked[i].value * 100).toDouble(),
    };

    // Store results
    _name = _labels[ranked.first.key];
    _accuracy = (ranked.first.value * 100).toDouble();
    _predictions = topResults;
  }

  /// Returns the top predicted label.
  String get name => _name;

  /// Returns confidence (accuracy %) of the top prediction.
  double get accuracy => _accuracy;

  /// Returns all top predictions as a map.
  Map<String, double> get predictions => _predictions;

  /// Returns whether the model is ready for use.
  bool get isReady => _ready;
}
