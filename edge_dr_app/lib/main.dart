// EDGE-DR Flutter App — main.dart
// Entry point with multilingual support and routing

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:provider/provider.dart';

import 'screens/home_screen.dart';
import 'screens/camera_screen.dart';
import 'screens/result_screen.dart';
import 'screens/history_screen.dart';
import 'screens/settings_screen.dart';
import 'services/inference_service.dart';
import 'services/database_service.dart';
import 'models/app_state.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock to portrait for clinical use
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);

  // Initialize services
  final dbService    = DatabaseService();
  await dbService.initialize();

  final inferenceService = InferenceService();
  await inferenceService.loadModel();

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AppState()),
        Provider<InferenceService>.value(value: inferenceService),
        Provider<DatabaseService>.value(value: dbService),
      ],
      child: const EdgeDRApp(),
    ),
  );
}

class EdgeDRApp extends StatelessWidget {
  const EdgeDRApp({super.key});

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();

    return MaterialApp(
      title: 'EDGE-DR',
      debugShowCheckedModeBanner: false,

      // ── Localisation ──────────────────────────
      locale: appState.locale,
      supportedLocales: const [
        Locale('en'),
        Locale('hi'),   // Hindi
        Locale('ta'),   // Tamil
        Locale('te'),   // Telugu
        Locale('mr'),   // Marathi
      ],
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],

      // ── Theme ─────────────────────────────────
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1565C0),   // deep blue — medical trust
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF1565C0),
          foregroundColor: Colors.white,
          elevation: 0,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF1565C0),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ),

      // ── Routes ────────────────────────────────
      initialRoute: '/',
      routes: {
        '/':        (_) => const HomeScreen(),
        '/camera':  (_) => const CameraScreen(),
        '/result':  (_) => const ResultScreen(),
        '/history': (_) => const HistoryScreen(),
        '/settings':(_) => const SettingsScreen(),
      },
    );
  }
}

