import 'package:flutter_test/flutter_test.dart';
import 'package:lite_vision_ai/lite_vision_ai.dart';

void main() {
  test('LiteVisionAI initial state test', () {
    final vision = LiteVisionAI();
    expect(vision.isReady, false);
    expect(vision.name, 'No analysis performed yet');
  });
}
