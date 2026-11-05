// EDGE-DR — History Screen
// View all past screening records stored in SQLite

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';

import '../services/database_service.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<Map<String, dynamic>> _records = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadRecords();
  }

  Future<void> _loadRecords() async {
    final db = context.read<DatabaseService>();
    final records = await db.getAllScreenings();
    if (mounted) {
      setState(() {
        _records   = records;
        _isLoading = false;
      });
    }
  }

  Color _gradeColor(int grade) {
    const colors = [
      Color(0xFF1976D2), Color(0xFF388E3C), Color(0xFFF9A825),
      Color(0xFFF57F17), Color(0xFFD32F2F),
    ];
    return grade < colors.length ? colors[grade] : Colors.grey;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Screening History',
            style: GoogleFonts.inter(fontWeight: FontWeight.bold)),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              setState(() => _isLoading = true);
              _loadRecords();
            },
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _records.isEmpty
              ? _buildEmpty()
              : ListView.separated(
                  padding: const EdgeInsets.all(16),
                  itemCount: _records.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 8),
                  itemBuilder: (_, i) => _buildRecordCard(_records[i]),
                ),
    );
  }

  Widget _buildEmpty() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.history, size: 64, color: Colors.grey.shade300),
          const SizedBox(height: 16),
          Text('No screenings yet',
              style: GoogleFonts.inter(fontSize: 18, color: Colors.grey)),
          const SizedBox(height: 8),
          Text('Tap "Start Screening" on the home screen',
              style: GoogleFonts.inter(fontSize: 13, color: Colors.grey)),
        ],
      ),
    );
  }

  Widget _buildRecordCard(Map<String, dynamic> record) {
    final grade      = record['grade'] as int;
    final gradeLabel = record['grade_label'] as String;
    final confidence = ((record['confidence'] as double) * 100).toStringAsFixed(1);
    final refer      = (record['refer'] as int) == 1;
    final dateStr    = DateFormat('dd MMM yyyy, HH:mm').format(
        DateTime.parse(record['created_at'] as String));
    final synced     = (record['synced'] as int) == 1;

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            // Grade badge
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: _gradeColor(grade),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Center(
                child: Text('$grade',
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold)),
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(gradeLabel,
                      style: GoogleFonts.inter(
                          fontWeight: FontWeight.bold, fontSize: 14)),
                  const SizedBox(height: 2),
                  Text('Confidence: $confidence%  •  $dateStr',
                      style: GoogleFonts.inter(
                          fontSize: 11, color: Colors.grey)),
                ],
              ),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                if (refer)
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: const Color(0xFFD32F2F).withOpacity(0.1),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(color: const Color(0xFFD32F2F)),
                    ),
                    child: const Text('REFER',
                        style: TextStyle(
                            color: Color(0xFFD32F2F),
                            fontSize: 10,
                            fontWeight: FontWeight.bold)),
                  ),
                const SizedBox(height: 4),
                Icon(synced ? Icons.cloud_done : Icons.cloud_off,
                    size: 16,
                    color: synced ? Colors.green : Colors.grey),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

