// EDGE-DR — Result Screen
// Clinical Triage Card shown to ASHA worker after inference

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import 'package:path_provider/path_provider.dart';
import 'package:intl/intl.dart';

import '../models/app_state.dart';
import '../models/triage_result.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final result = context.watch<AppState>().currentResult as TriageResult?;

    if (result == null) {
      return const Scaffold(
        body: Center(child: Text('No result to display.')),
      );
    }

    final urgencyColor = Color(result.urgencyColorValue);

    return Scaffold(
      appBar: AppBar(
        title: Text('Screening Result',
            style: GoogleFonts.inter(fontWeight: FontWeight.bold)),
        actions: [
          IconButton(
            icon: const Icon(Icons.picture_as_pdf),
            tooltip: 'Generate Referral PDF',
            onPressed: () => _generateReferralPDF(context, result),
          ),
          IconButton(
            icon: const Icon(Icons.share),
            tooltip: 'Share Result',
            onPressed: () => _shareResult(result),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // ── Verdict card ─────────────────────
            _buildVerdictCard(result, urgencyColor),
            const SizedBox(height: 16),

            // ── Heatmap / image ──────────────────
            if (result.imagePath.isNotEmpty) _buildImageCard(result),
            const SizedBox(height: 16),

            // ── Confidence chart ─────────────────
            _buildConfidenceChart(result),
            const SizedBox(height: 16),

            // ── Vernacular summary ───────────────
            _buildVernacularCard(result),
            const SizedBox(height: 16),

            // ── Action button ────────────────────
            if (result.refer) _buildReferButton(context, result, urgencyColor),

            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  // ────────────────────────────────────────────
  Widget _buildVerdictCard(TriageResult result, Color urgencyColor) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: urgencyColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: urgencyColor, width: 2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(result.refer ? Icons.warning_amber_rounded : Icons.check_circle,
                  color: urgencyColor, size: 36),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('DR Grade ${result.grade}',
                        style: GoogleFonts.inter(
                            color: urgencyColor,
                            fontSize: 13,
                            fontWeight: FontWeight.w600)),
                    Text(result.gradeLabel,
                        style: GoogleFonts.inter(
                            fontSize: 20, fontWeight: FontWeight.bold)),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(result.confidencePercent,
                      style: GoogleFonts.inter(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: urgencyColor)),
                  Text('confidence',
                      style: GoogleFonts.inter(
                          fontSize: 11, color: Colors.grey)),
                ],
              ),
            ],
          ),
          const SizedBox(height: 16),
          const Divider(),
          const SizedBox(height: 8),
          Text('Recommended Action',
              style: GoogleFonts.inter(fontSize: 12, color: Colors.grey)),
          const SizedBox(height: 4),
          Text(result.triageAction,
              style: GoogleFonts.inter(
                  fontSize: 15, fontWeight: FontWeight.bold,
                  color: urgencyColor)),
          const SizedBox(height: 8),
          Row(
            children: [
              const Icon(Icons.timer, size: 14, color: Colors.grey),
              const SizedBox(width: 4),
              Text('Inference: ${result.latencyMs} ms',
                  style: GoogleFonts.inter(fontSize: 11, color: Colors.grey)),
            ],
          ),
        ],
      ),
    );
  }

  // ────────────────────────────────────────────
  Widget _buildImageCard(TriageResult result) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(color: Colors.black12, blurRadius: 8, offset: const Offset(0, 3))
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: result.heatmapPath != null && File(result.heatmapPath!).existsSync()
            ? Image.file(File(result.heatmapPath!),
                fit: BoxFit.contain,
                errorBuilder: (_, __, ___) => _rawImage(result))
            : _rawImage(result),
      ),
    );
  }

  Widget _rawImage(TriageResult result) {
    return Image.file(File(result.imagePath), fit: BoxFit.contain);
  }

  // ────────────────────────────────────────────
  Widget _buildConfidenceChart(TriageResult result) {
    const labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'];
    const colors = [
      Color(0xFF1976D2), Color(0xFF388E3C), Color(0xFFF9A825),
      Color(0xFFF57F17), Color(0xFFD32F2F),
    ];

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 6)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Grade Confidence Distribution',
              style: GoogleFonts.inter(
                  fontWeight: FontWeight.bold, fontSize: 14)),
          const SizedBox(height: 16),
          SizedBox(
            height: 150,
            child: BarChart(
              BarChartData(
                alignment: BarChartAlignment.spaceAround,
                maxY: 1.0,
                barGroups: List.generate(5, (i) {
                  return BarChartGroupData(
                    x: i,
                    barRods: [
                      BarChartRodData(
                        toY: result.classProbs[i],
                        color: i == result.grade
                            ? Color(result.urgencyColorValue)
                            : colors[i].withOpacity(0.5),
                        width: 32,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ],
                  );
                }),
                titlesData: FlTitlesData(
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      getTitlesWidget: (v, _) => Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(labels[v.toInt()],
                            style: const TextStyle(fontSize: 9)),
                      ),
                    ),
                  ),
                  leftTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  topTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  rightTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                ),
                gridData: const FlGridData(show: false),
                borderData: FlBorderData(show: false),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ────────────────────────────────────────────
  Widget _buildVernacularCard(TriageResult result) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF8E1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFF9A825)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.translate, color: Color(0xFFF57F17)),
              const SizedBox(width: 8),
              Text('Patient Summary',
                  style: GoogleFonts.inter(fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 12),
          _langRow('🇮🇳 हिन्दी', result.hindiSummary),
          const SizedBox(height: 8),
          _langRow('🏳 தமிழ்', result.tamilSummary),
          const SizedBox(height: 8),
          _langRow('🏳 తెలుగు', result.teluguSummary),
        ],
      ),
    );
  }

  Widget _langRow(String label, String text) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            style: const TextStyle(fontSize: 11, color: Colors.grey,
                fontWeight: FontWeight.bold)),
        const SizedBox(height: 2),
        Text(text, style: const TextStyle(fontSize: 14)),
      ],
    );
  }

  // ────────────────────────────────────────────
  Widget _buildReferButton(BuildContext ctx, TriageResult result, Color color) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          padding: const EdgeInsets.symmetric(vertical: 18),
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14)),
        ),
        icon: const Icon(Icons.local_hospital),
        label: const Text('Generate Referral Ticket',
            style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold)),
        onPressed: () => _generateReferralPDF(ctx, result),
      ),
    );
  }

  // ────────────────────────────────────────────
  // PDF referral ticket
  // ────────────────────────────────────────────

  Future<void> _generateReferralPDF(
      BuildContext context, TriageResult result) async {
    final pdf = pw.Document();
    final now = DateFormat('dd MMM yyyy, HH:mm').format(result.timestamp);

    pdf.addPage(pw.Page(
      pageFormat: PdfPageFormat.a4,
      build: (pw.Context ctx) {
        return pw.Column(
          crossAxisAlignment: pw.CrossAxisAlignment.start,
          children: [
            pw.Header(level: 0, child: pw.Text('EDGE-DR Referral Ticket',
                style: pw.TextStyle(fontSize: 22, fontWeight: pw.FontWeight.bold))),
            pw.SizedBox(height: 20),
            pw.Text('Date / Time: $now'),
            pw.Text('DR Grade: ${result.grade} — ${result.gradeLabel}'),
            pw.Text('AI Confidence: ${result.confidencePercent}'),
            pw.SizedBox(height: 10),
            pw.Divider(),
            pw.SizedBox(height: 10),
            pw.Text('Recommended Action:',
                style: pw.TextStyle(fontWeight: pw.FontWeight.bold)),
            pw.Text(result.triageAction,
                style: pw.TextStyle(fontSize: 14)),
            pw.SizedBox(height: 20),
            pw.Text('Patient Summary (Hindi):'),
            pw.Text(result.hindiSummary),
            pw.SizedBox(height: 20),
            pw.Text('Generated by EDGE-DR v1.0 | 100% offline AI screening',
                style: pw.TextStyle(
                    fontSize: 10, color: PdfColors.grey)),
          ],
        );
      },
    ));

    await Printing.layoutPdf(
      onLayout: (format) async => pdf.save(),
    );
  }

  Future<void> _shareResult(TriageResult result) async {
    final text = '''
EDGE-DR Screening Result
Grade: ${result.grade} — ${result.gradeLabel}
Confidence: ${result.confidencePercent}
Action: ${result.triageAction}
${result.hindiSummary}
''';
    await Share.share(text, subject: 'EDGE-DR Screening Report');
  }
}

