// EDGE-DR — TFLite Inference Service
// Loads the INT8 model, runs preprocessing, returns TriageResult

import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:logger/logger.dart';

import '../models/triage_result.dart';

final _log = Logger(printer: PrettyPrinter(methodCount: 0));

// ImageNet normalisation constants
const _mean = [0.485, 0.456, 0.406];
const _std  = [0.229, 0.224, 0.225];

const _modelInputSize = 224;

const _gradeLabels = [
  'No Diabetic Retinopathy',
  'Mild NPDR',
  'Moderate NPDR',
  'Severe NPDR',
  'Proliferative DR (PDR)',
];

const _referralGrades = {2, 3, 4};

// ──────────────────────────────────────────────
// Localized summaries
// ──────────────────────────────────────────────
const _hindiSummary = [
  'कोई डायबिटिक रेटिनोपैथी नहीं मिली। वार्षिक जाँच जारी रखें।',
  'हल्की डायबिटिक रेटिनोपैथी मिली है। 6-12 महीनों में पुनः जाँच कराएँ।',
  'मध्यम डायबिटिक रेटिनोपैथी मिली है। कृपया नेत्र चिकित्सक से मिलें।',
  'गंभीर रेटिनोपैथी मिली है। तुरंत नेत्र चिकित्सक से मिलें।',
  'बहुत गंभीर रेटिनोपैथी मिली है। 48 घंटों में नेत्र विशेषज्ञ से मिलें।',
];

const _tamilSummary = [
  'நீரிழிவு விழித்திரை நோய் இல்லை. ஆண்டு பரிசோதனை தொடரவும்.',
  'மென்மையான நோய் கண்டறியப்பட்டது. 6-12 மாதங்களில் மறுபரிசோதனை.',
  'மிதமான நோய் உள்ளது. 4-6 வாரங்களுக்குள் கண் மருத்துவரை அணுகவும்.',
  'தீவிர நோய் உள்ளது. 1-2 வாரங்களுக்குள் கண் மருத்துவரை அணுகவும்.',
  'மிகவும் தீவிரமான நோய். 48 மணி நேரத்தில் நிபுணரை அணுகவும்.',
];

const _teluguSummary = [
  'డయాబెటిక్ రెటినోపతి లేదు. వార్షిక స్క్రీనింగ్ కొనసాగించండి.',
  'తేలికపాటి వ్యాధి. 6-12 నెలల్లో తిరిగి తనిఖీ చేయించుకోండి.',
  'మధ్యస్థ వ్యాధి. 4-6 వారాల్లో కంటి వైద్యుడిని సందర్శించండి.',
  'తీవ్రమైన వ్యాధి. 1-2 వారాల్లో కంటి వైద్యుడిని సందర్శించండి.',
  'చాలా తీవ్రమైన వ్యాధి. 48 గంటల్లో నిపుణ వైద్యుడిని సందర్శించండి.',
];


class InferenceService {
  Interpreter? _interpreter;
  bool _isLoaded = false;

  /// Load TFLite INT8 model from assets
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/edge_dr.tflite',
        options: InterpreterOptions()
          ..threads = 4
          ..useNnApiForAndroid = true,   // Android NNAPI / NPU acceleration
      );
      _isLoaded = true;
      _log.i('TFLite model loaded. '
             'Input: ${_interpreter!.getInputTensor(0).shape} '
             'Output: ${_interpreter!.getOutputTensor(0).shape}');
    } catch (e) {
      _log.e('Failed to load TFLite model: $e');
      rethrow;
    }
  }

  bool get isLoaded => _isLoaded;

  // ──────────────────────────────────────────────
  // Main predict function
  // ──────────────────────────────────────────────

  /// Run full inference on a File image. Returns [TriageResult].
  Future<TriageResult> predict(File imageFile) async {
    if (!_isLoaded || _interpreter == null) {
      throw StateError('Model not loaded. Call loadModel() first.');
    }

    final stopwatch = Stopwatch()..start();

    // 1. Load & preprocess image
    final inputTensor = await _preprocessImage(imageFile);

    // 2. Run TFLite inference
    final output = List.filled(5, 0.0).reshape([1, 5]);
    _interpreter!.run(inputTensor, output);

    // 3. Softmax + argmax
    final logits   = List<double>.from(output[0]);
    final probs    = _softmax(logits);
    final grade    = _argmax(probs);
    final confidence = probs[grade];

    stopwatch.stop();
    final latencyMs = stopwatch.elapsedMilliseconds;

    _log.d('Inference complete: grade=$grade '
           'conf=${(confidence * 100).toStringAsFixed(1)}% '
           'latency=${latencyMs}ms');

    // 4. Build TriageResult
    return TriageResult(
      grade          : grade,
      gradeLabel     : _gradeLabels[grade],
      confidence     : confidence,
      classProbs     : probs,
      refer          : _referralGrades.contains(grade),
      triageAction   : _triageAction(grade),
      urgency        : _urgency(grade),
      hindiSummary   : _hindiSummary[grade],
      tamilSummary   : _tamilSummary[grade],
      teluguSummary  : _teluguSummary[grade],
      latencyMs      : latencyMs,
      imagePath      : imageFile.path,
      timestamp      : DateTime.now(),
    );
  }

  // ──────────────────────────────────────────────
  // Preprocessing (matches Python pipeline)
  // ──────────────────────────────────────────────

  Future<List<List<List<List<double>>>>> _preprocessImage(File file) async {
    final bytes = await file.readAsBytes();
    img.Image? image = img.decodeImage(bytes);
    if (image == null) throw Exception('Cannot decode image: ${file.path}');

    // Resize to model input size
    image = img.copyResize(image,
        width: _modelInputSize, height: _modelInputSize,
        interpolation: img.Interpolation.linear);

    // Apply circular mask (approximate with center crop — hardware lens already crops)
    // Apply CLAHE-equivalent: histogram equalise on L channel (approximate)
    image = _applyCLAHEApprox(image);

    // Build [1, 224, 224, 3] float tensor with ImageNet normalisation
    final input = List.generate(
      1, (_) => List.generate(
        _modelInputSize, (y) => List.generate(
          _modelInputSize, (x) {
            final pixel = image!.getPixel(x, y);
            return [
              ((pixel.r / 255.0) - _mean[0]) / _std[0],
              ((pixel.g / 255.0) - _mean[1]) / _std[1],
              ((pixel.b / 255.0) - _mean[2]) / _std[2],
            ];
          },
        ),
      ),
    );

    return input;
  }

  /// Approximate CLAHE: simple contrast stretch per channel
  img.Image _applyCLAHEApprox(img.Image src) {
    // This is a lightweight approximation — real CLAHE runs in preprocess.py
    return img.adjustColor(src, contrast: 1.3, brightness: 1.05);
  }

  // ──────────────────────────────────────────────
  // Helpers
  // ──────────────────────────────────────────────

  List<double> _softmax(List<double> logits) {
    final maxL = logits.reduce(math.max);
    final exps = logits.map((l) => math.exp(l - maxL)).toList();
    final sum  = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sum).toList();
  }

  int _argmax(List<double> values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }

  String _triageAction(int grade) {
    const actions = [
      'MONITOR: Annual screening recommended',
      'MONITOR: Recheck in 6–12 months',
      'REFER: District Eye Hospital within 4–6 weeks',
      'REFER URGENTLY: Within 1–2 weeks',
      'REFER IMMEDIATELY: Within 48–72 hours — vision at risk',
    ];
    return actions[grade];
  }

  String _urgency(int grade) {
    const urgencies = ['none', 'low', 'moderate', 'high', 'critical'];
    return urgencies[grade];
  }

  void dispose() {
    _interpreter?.close();
  }
}

