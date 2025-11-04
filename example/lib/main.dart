import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:lite_vision_ai/lite_vision_ai.dart';

/// ---------------------------------------------------------------------------
/// 📱 LiteVision AI Example
/// A minimal demo app that shows how to use the LiteVisionAI package
/// for TensorFlow Lite image classification.
/// ---------------------------------------------------------------------------

void main() {
  runApp(const MaterialApp(
    debugShowCheckedModeBanner: false,
    home: LiteVisionDemo(),
  ));
}

class LiteVisionDemo extends StatefulWidget {
  const LiteVisionDemo({super.key});

  @override
  State<LiteVisionDemo> createState() => _LiteVisionDemoState();
}

class _LiteVisionDemoState extends State<LiteVisionDemo> {
  final vision = LiteVisionAI();
  final picker = ImagePicker();
  bool ready = false;
  File? imageFile;

  @override
  void initState() {
    super.initState();
    _initVision();
  }

  /// Load model and labels at startup
  Future<void> _initVision() async {
    await vision.load(
      model: 'assets/models/model.tflite',
      labels: 'assets/models/labels.txt',
    );
    setState(() => ready = true);
  }

  /// Select image from gallery and classify it
  Future<void> _pickAndClassify() async {
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;
    final file = File(picked.path);
    setState(() => imageFile = file);

    await vision.classify(image: file, top: 3);
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("LiteVision AI Demo")),
      body: Center(
        child: ready
            ? Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: _pickAndClassify,
                    child: const Text("Pick Image"),
                  ),
                  const SizedBox(height: 20),
                  if (imageFile != null) Image.file(imageFile!, height: 200),
                  const SizedBox(height: 10),
                  if (vision.name != "No analysis performed yet") ...[
                    Text("📸 Label: ${vision.name}"),
                    Text("Accuracy: ${vision.accuracy.toStringAsFixed(2)}%"),
                    const SizedBox(height: 10),
                    const Text("Top Predictions:"),
                    ...vision.predictions.entries.map(
                      (e) => Text("${e.key}: ${e.value.toStringAsFixed(2)}%"),
                    ),
                  ],
                ],
              )
            : const CircularProgressIndicator(),
      ),
    );
  }
}
