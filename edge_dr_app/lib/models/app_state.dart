// EDGE-DR — App State (Provider)
import 'package:flutter/material.dart';

class AppState extends ChangeNotifier {
  // Language preference
  Locale _locale = const Locale('en');
  Locale get locale => _locale;

  void setLocale(Locale locale) {
    _locale = locale;
    notifyListeners();
  }

  // Current screening result (passed between Camera → Result screens)
  dynamic _currentResult;
  dynamic get currentResult => _currentResult;

  void setCurrentResult(dynamic result) {
    _currentResult = result;
    notifyListeners();
  }

  // Loading state
  bool _isInferring = false;
  bool get isInferring => _isInferring;

  void setInferring(bool v) {
    _isInferring = v;
    notifyListeners();
  }
}

