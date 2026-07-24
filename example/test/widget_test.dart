import 'package:flutter_test/flutter_test.dart';
import 'package:lite_vision_ai_example/main.dart';

void main() {
  testWidgets('Verify LiteVisionDemo loads', (WidgetTester tester) async {
    await tester.pumpWidget(const LiteVisionDemo());
    expect(find.text('LiteVision AI Demo'), findsOneWidget);
  });
}
