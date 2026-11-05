// EDGE-DR — TriageResult model
// Plain-data class representing the output of one screening session

class TriageResult {
  final int           grade;
  final String        gradeLabel;
  final double        confidence;        // 0.0–1.0
  final List<double>  classProbs;        // [5] softmax probabilities
  final bool          refer;
  final String        triageAction;
  final String        urgency;           // none | low | moderate | high | critical
  final String        hindiSummary;
  final String        tamilSummary;
  final String        teluguSummary;
  final int           latencyMs;
  final String        imagePath;
  final DateTime      timestamp;

  // Set after DB insert
  String? recordId;
  String? heatmapPath;

  TriageResult({
    required this.grade,
    required this.gradeLabel,
    required this.confidence,
    required this.classProbs,
    required this.refer,
    required this.triageAction,
    required this.urgency,
    required this.hindiSummary,
    required this.tamilSummary,
    required this.teluguSummary,
    required this.latencyMs,
    required this.imagePath,
    required this.timestamp,
  });

  /// Urgency colour for UI
  int get urgencyColorValue {
    switch (urgency) {
      case 'critical': return 0xFFD32F2F;   // red
      case 'high':     return 0xFFF57F17;   // amber
      case 'moderate': return 0xFFF9A825;   // yellow
      case 'low':      return 0xFF388E3C;   // green
      default:         return 0xFF1976D2;   // blue
    }
  }

  String get confidencePercent =>
      '${(confidence * 100).toStringAsFixed(1)}%';

  Map<String, dynamic> toMap() => {
    'grade':          grade,
    'grade_label':    gradeLabel,
    'confidence':     confidence,
    'refer':          refer,
    'triage_action':  triageAction,
    'urgency':        urgency,
    'latency_ms':     latencyMs,
    'image_path':     imagePath,
    'timestamp':      timestamp.toIso8601String(),
  };
}

