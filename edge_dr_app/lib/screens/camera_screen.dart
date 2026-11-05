// EDGE-DR — Camera Screen
// Capture fundus image and trigger inference

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shimmer/shimmer.dart';

import '../models/app_state.dart';
import '../models/triage_result.dart';
import '../services/inference_service.dart';
import '../services/database_service.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  File?       _selectedImage;
  String?     _statusMessage;
  bool        _isProcessing = false;

  final _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    final xFile = await _picker.pickImage(
      source: source,
      imageQuality: 90,
      maxWidth: 1024,
      maxHeight: 1024,
    );
    if (xFile == null) return;
    setState(() {
      _selectedImage = File(xFile.path);
      _statusMessage = 'Image loaded. Tap Analyse to continue.';
    });
  }

  Future<void> _runInference() async {
    if (_selectedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please capture or select an image first.')),
      );
      return;
    }

    setState(() {
      _isProcessing  = true;
      _statusMessage = 'Running AI analysis…';
    });
    context.read<AppState>().setInferring(true);

    try {
      final inferenceService = context.read<InferenceService>();
      final dbService        = context.read<DatabaseService>();

      final TriageResult result = await inferenceService.predict(_selectedImage!);

      // Save to local SQLite
      final recordId = await dbService.insertScreening(result: result);
      result.recordId = recordId;

      if (mounted) {
        context.read<AppState>().setCurrentResult(result);
        context.read<AppState>().setInferring(false);
        Navigator.pushNamed(context, '/result');
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _statusMessage = 'Error during analysis: $e';
          _isProcessing  = false;
        });
        context.read<AppState>().setInferring(false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Capture Fundus Image',
            style: GoogleFonts.inter(fontWeight: FontWeight.bold)),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            // ── Image preview ───────────────────
            Expanded(
              child: _selectedImage == null
                  ? _buildPlaceholder()
                  : _buildImagePreview(),
            ),
            const SizedBox(height: 16),

            // ── Status message ──────────────────
            if (_statusMessage != null)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFFE3F2FD),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.info_outline,
                        color: Color(0xFF1565C0), size: 18),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(_statusMessage!,
                          style: const TextStyle(color: Color(0xFF1565C0))),
                    ),
                  ],
                ),
              ),
            const SizedBox(height: 16),

            // ── Capture buttons ─────────────────
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                    onPressed: _isProcessing
                        ? null
                        : () => _pickImage(ImageSource.camera),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                    onPressed: _isProcessing
                        ? null
                        : () => _pickImage(ImageSource.gallery),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // ── Analyse button ──────────────────
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                icon: _isProcessing
                    ? const SizedBox(
                        width: 20, height: 20,
                        child: CircularProgressIndicator(
                            color: Colors.white, strokeWidth: 2))
                    : const Icon(Icons.psychology),
                label: Text(
                  _isProcessing ? 'Analysing…' : 'Analyse Image',
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 18),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14)),
                ),
                onPressed: _isProcessing ? null : _runInference,
              ),
            ),

            const SizedBox(height: 8),

            // Quality hint
            Text(
              'Tip: Ensure good lighting. Hold camera steady 2–3 cm from eye.',
              style: GoogleFonts.inter(fontSize: 11, color: Colors.grey),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPlaceholder() {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFFF5F7FA),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey.shade300, width: 2),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.remove_red_eye_outlined,
              size: 80, color: Colors.grey.shade400),
          const SizedBox(height: 16),
          Text('No image selected',
              style: GoogleFonts.inter(
                  fontSize: 16, color: Colors.grey.shade500)),
          const SizedBox(height: 8),
          Text('Use Camera or Gallery below',
              style: GoogleFonts.inter(
                  fontSize: 13, color: Colors.grey.shade400)),
        ],
      ),
    );
  }

  Widget _buildImagePreview() {
    return Stack(
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Image.file(
            _selectedImage!,
            fit: BoxFit.contain,
            width: double.infinity,
          ),
        ),
        if (_isProcessing)
          Positioned.fill(
            child: Shimmer.fromColors(
              baseColor: Colors.black26,
              highlightColor: Colors.white24,
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black26,
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
            ),
          ),
      ],
    );
  }
}

