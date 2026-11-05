// EDGE-DR — Settings Screen
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../models/app_state.dart';
import '../services/database_service.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Settings',
            style: GoogleFonts.inter(fontWeight: FontWeight.bold)),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // ── Language ─────────────────────────
          _sectionHeader('Display Language'),
          _buildLanguageTile(context, 'English', 'en'),
          _buildLanguageTile(context, 'हिन्दी (Hindi)', 'hi'),
          _buildLanguageTile(context, 'தமிழ் (Tamil)', 'ta'),
          _buildLanguageTile(context, 'తెలుగు (Telugu)', 'te'),
          _buildLanguageTile(context, 'मराठी (Marathi)', 'mr'),

          const SizedBox(height: 24),

          // ── About ─────────────────────────────
          _sectionHeader('About'),
          ListTile(
            leading: const Icon(Icons.info_outline),
            title: const Text('EDGE-DR v1.0'),
            subtitle: const Text('Offline Explainable AI — Diabetic Retinopathy Screening'),
          ),
          ListTile(
            leading: const Icon(Icons.memory),
            title: const Text('AI Model'),
            subtitle: const Text('EfficientNet-B0 INT8 TFLite · <12 MB · <1.2s latency'),
          ),
          ListTile(
            leading: const Icon(Icons.wifi_off),
            title: const Text('Connectivity'),
            subtitle: const Text('100% offline — no internet required for screening'),
          ),

          const SizedBox(height: 24),

          // ── Danger zone ───────────────────────
          _sectionHeader('Data Management'),
          ListTile(
            leading: const Icon(Icons.delete_forever, color: Colors.red),
            title: const Text('Clear All Records',
                style: TextStyle(color: Colors.red)),
            subtitle: const Text('Permanently delete all local screening data'),
            onTap: () => _confirmClear(context),
          ),
        ],
      ),
    );
  }

  Widget _sectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Text(title.toUpperCase(),
          style: GoogleFonts.inter(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              color: Colors.grey,
              letterSpacing: 1.2)),
    );
  }

  Widget _buildLanguageTile(BuildContext ctx, String label, String code) {
    final current = ctx.watch<AppState>().locale.languageCode;
    return RadioListTile<String>(
      title: Text(label),
      value: code,
      groupValue: current,
      onChanged: (v) {
        if (v != null) ctx.read<AppState>().setLocale(Locale(v));
      },
    );
  }

  Future<void> _confirmClear(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Clear All Records?'),
        content: const Text(
            'This will permanently delete all local screening history. '
            'This cannot be undone.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete All',
                style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
    if (confirmed == true && context.mounted) {
      await context.read<DatabaseService>().clearAll();
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('All records deleted.')),
      );
    }
  }
}

