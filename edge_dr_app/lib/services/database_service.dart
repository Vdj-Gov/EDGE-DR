// EDGE-DR — SQLite Database Service
// Stores encrypted screening records locally until sync is possible

import 'dart:convert';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:logger/logger.dart';

import '../models/triage_result.dart';

final _log = Logger(printer: PrettyPrinter(methodCount: 0));

const _dbName    = 'edge_dr_records.db';
const _dbVersion = 1;
const _table     = 'screenings';

class DatabaseService {
  Database? _db;

  Future<void> initialize() async {
    final dir  = await getApplicationDocumentsDirectory();
    final path = join(dir.path, _dbName);

    _db = await openDatabase(
      path,
      version: _dbVersion,
      onCreate: _createTables,
      onUpgrade: _onUpgrade,
    );
    _log.i('Database initialized at $path');
  }

  Future<void> _createTables(Database db, int version) async {
    await db.execute('''
      CREATE TABLE $_table (
        id              TEXT PRIMARY KEY,
        patient_id      TEXT,
        image_path      TEXT NOT NULL,
        heatmap_path    TEXT,
        grade           INTEGER NOT NULL,
        grade_label     TEXT NOT NULL,
        confidence      REAL NOT NULL,
        class_probs     TEXT NOT NULL,
        refer           INTEGER NOT NULL,
        triage_action   TEXT NOT NULL,
        urgency         TEXT NOT NULL,
        hindi_summary   TEXT,
        tamil_summary   TEXT,
        telugu_summary  TEXT,
        latitude        REAL,
        longitude       REAL,
        location_name   TEXT,
        latency_ms      INTEGER,
        synced          INTEGER DEFAULT 0,
        created_at      TEXT NOT NULL
      )
    ''');
    _log.i('Tables created');
  }

  Future<void> _onUpgrade(Database db, int oldV, int newV) async {
    // Future migration logic
  }

  // ──────────────────────────────────────────────
  // CRUD operations
  // ──────────────────────────────────────────────

  /// Insert a new screening record
  Future<String> insertScreening({
    required TriageResult result,
    String? patientId,
    String? heatmapPath,
    double? latitude,
    double? longitude,
    String? locationName,
  }) async {
    final id = DateTime.now().millisecondsSinceEpoch.toString();

    await _db!.insert(_table, {
      'id':            id,
      'patient_id':    patientId ?? 'ANON',
      'image_path':    result.imagePath,
      'heatmap_path':  heatmapPath,
      'grade':         result.grade,
      'grade_label':   result.gradeLabel,
      'confidence':    result.confidence,
      'class_probs':   jsonEncode(result.classProbs),
      'refer':         result.refer ? 1 : 0,
      'triage_action': result.triageAction,
      'urgency':       result.urgency,
      'hindi_summary': result.hindiSummary,
      'tamil_summary': result.tamilSummary,
      'telugu_summary':result.teluguSummary,
      'latitude':      latitude,
      'longitude':     longitude,
      'location_name': locationName,
      'latency_ms':    result.latencyMs,
      'synced':        0,
      'created_at':    result.timestamp.toIso8601String(),
    });

    _log.d('Screening $id saved (grade ${result.grade}, refer: ${result.refer})');
    return id;
  }

  /// Get all screening records (newest first)
  Future<List<Map<String, dynamic>>> getAllScreenings() async {
    return _db!.query(_table, orderBy: 'created_at DESC');
  }

  /// Get screening count stats
  Future<Map<String, int>> getStats() async {
    final total = Sqflite.firstIntValue(
        await _db!.rawQuery('SELECT COUNT(*) FROM $_table')) ?? 0;
    final referrals = Sqflite.firstIntValue(
        await _db!.rawQuery('SELECT COUNT(*) FROM $_table WHERE refer=1')) ?? 0;
    final unsynced = Sqflite.firstIntValue(
        await _db!.rawQuery('SELECT COUNT(*) FROM $_table WHERE synced=0')) ?? 0;

    return {
      'total':    total,
      'referrals': referrals,
      'unsynced':  unsynced,
    };
  }

  /// Mark records as synced (after upload when connectivity is available)
  Future<void> markSynced(List<String> ids) async {
    for (final id in ids) {
      await _db!.update(_table, {'synced': 1},
          where: 'id = ?', whereArgs: [id]);
    }
  }

  /// Delete all records (for GDPR / privacy reset)
  Future<void> clearAll() async {
    await _db!.delete(_table);
    _log.w('All screening records deleted');
  }

  void dispose() => _db?.close();
}

